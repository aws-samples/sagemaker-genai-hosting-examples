# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Stage everything the endpoint needs, so it can run with network isolation.

The default notebook lets the container download the Decider runtime and the model
weights at startup. A network-isolated container cannot reach PyPI, GitHub, or Hugging
Face, so this module stages them ahead of time into one directory that SageMaker AI
downloads from Amazon S3 before the container starts:

    code/inference.py           the same adapter as the default deployment
    code/requirements.txt       installs only from code/wheels (no index)
    code/wheels/                the Decider runtime and huggingface_hub wheels (the serving
                                image provides PyTorch and the other libraries)
    code/provenance.json        pinned source URL and SHA-256 of the bundled Decider wheel
    hf-cache/                   a Hugging Face cache with the pinned checkpoint and base
"""

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

import huggingface_hub
from huggingface_hub import snapshot_download
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename

from inference import BASE_REVISION, MODEL_ID, MODEL_REVISION, SOURCE_URL

BASE_ID = "Qwen/Qwen3.5-2B-Base"
CHECKPOINT_FILES = [
    "hobson_config.json", "head.safetensors", "lora/*",
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "provenance.json", "MANIFEST.sha256", "LICENSE.md",
]
# The serving image (PyTorch 2.14 DLC on Amazon Linux 2023) runs CPython 3.12 on x86_64 Linux.
PLATFORM_ARGS = [
    "--only-binary=:all:", "--implementation", "cp", "--python-version", "3.12",
    "--abi", "cp312", "--platform", "manylinux_2_28_x86_64",
    "--platform", "manylinux_2_17_x86_64", "--platform", "manylinux2014_x86_64",
]
# pip evaluates environment markers such as platform_system == "Linux" against the machine
# it runs on, even with --platform. Evaluate them against the container instead.
TARGET_ENV = {
    **default_environment(), "os_name": "posix", "sys_platform": "linux",
    "platform_system": "Linux", "platform_machine": "x86_64",
    "implementation_name": "cpython", "platform_python_implementation": "CPython",
    "python_version": "3.12", "python_full_version": "3.12.0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pip(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", *args], check=True)


def missing_for_target(wheels: Path, top_level: list[str]) -> set[str]:
    """Requirements that apply in the container but have no matching wheel in `wheels`.

    Checks each requirement's version range as well as its name, so staging fails here
    rather than at offline install time.
    """
    available = {}  # name -> [(version, Requires-Dist lines)]
    for whl in wheels.glob("*.whl"):
        name, ver, _, _ = parse_wheel_filename(whl.name)
        with zipfile.ZipFile(whl) as archive:
            meta = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            requires = Parser().parsestr(archive.read(meta).decode()).get_all("Requires-Dist") or []
        available.setdefault(canonicalize_name(name), []).append((ver, requires))
    pending = [Requirement(line) for line in top_level]
    seen, extras, missing = set(), {}, set()
    while pending:
        req = pending.pop()
        key = canonicalize_name(req.name)
        if (key, str(req.specifier), frozenset(req.extras)) in seen:
            continue
        seen.add((key, str(req.specifier), frozenset(req.extras)))
        extras.setdefault(key, set()).update(req.extras)
        matches = [(v, r) for v, r in available.get(key, [])
                   if req.specifier.contains(v, prereleases=True)]
        if not matches:
            missing.add(f"{req.name}{req.specifier}")
            continue
        for line in max(matches)[1]:
            dep = Requirement(line)
            if not dep.marker or any(dep.marker.evaluate({**TARGET_ENV, "extra": e})
                                     for e in {"", *extras[key]}):
                pending.append(dep)
    return missing


def stage(example_dir: Path, stage_dir: Path) -> dict:
    """Build `stage_dir` from the example's inference.py and requirements.txt."""
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    code, wheels = stage_dir / "code", stage_dir / "code" / "wheels"
    wheels.mkdir(parents=True)
    shutil.copy(example_dir / "inference.py", code / "inference.py")

    # 1. Decider itself: build a wheel from the pinned source archive. Its dependencies
    #    (PyTorch, transformers, peft, ...) come from the serving image.
    pip("wheel", "--no-deps", "--quiet", "--wheel-dir", str(wheels), SOURCE_URL)
    decider_wheel = next(wheels.glob("strands_decider-*.whl"))
    decider_version = decider_wheel.name.split("-")[1]

    # 2. huggingface_hub at this notebook's version: the container reads the cache this
    #    notebook writes, and the cache layout differs across huggingface_hub versions.
    hub_pin = f"huggingface_hub=={huggingface_hub.__version__}"
    pip("download", "--quiet", "--no-deps", "--dest", str(wheels), *PLATFORM_ARGS, hub_pin)

    # 3. Any other pins in requirements.txt, resolved for the container's platform.
    lines = (example_dir / "requirements.txt").read_text().splitlines()
    index_args = [a for line in lines if line.startswith("--extra-index-url") for a in line.split()]
    pins = [line for line in lines if line.strip() and not line.startswith(("--", "#", "strands-decider"))]
    if pins:
        pip("download", "--quiet", "--dest", str(wheels), *PLATFORM_ARGS, *index_args, *pins)
        for _ in range(5):  # add Linux-only dependencies that pip skipped on this machine
            missing = missing_for_target(wheels, pins)
            if not missing:
                break
            pip("download", "--quiet", "--dest", str(wheels), *PLATFORM_ARGS, *index_args, *sorted(missing))
        else:
            raise RuntimeError(f"Could not resolve wheels for: {sorted(missing)}")

    # 4. Offline requirements: installed from code/wheels only, on top of the serving image.
    (code / "requirements.txt").write_text("\n".join(
        ["--no-index", "--find-links /opt/ml/model/code/wheels", *pins, hub_pin,
         f"strands-decider=={decider_version}", ""]))
    provenance = {"source_url": SOURCE_URL, "wheel": decider_wheel.name,
                  "version": decider_version, "sha256": sha256(decider_wheel)}
    (code / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    # 5. Weights: the pinned checkpoint and base revision as a Hugging Face cache, so the
    #    adapter's revision-pinned lookups resolve locally with HF_HUB_OFFLINE=1. S3 has no
    #    symlinks, so keep only each repo's snapshot (as plain files, moved rather than
    #    copied to keep peak disk near the final size) and, where this huggingface_hub
    #    version writes one, its cached file listing (trees/).
    download = stage_dir.parent / f"{stage_dir.name}-hf-download"
    try:
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=download, allow_patterns=CHECKPOINT_FILES)
        snapshot_download(BASE_ID, revision=BASE_REVISION, cache_dir=download)
        cache = stage_dir / "hf-cache"
        for repo in download.glob("models--*"):
            if (repo / "trees").is_dir():
                shutil.copytree(repo / "trees", cache / repo.name / "trees")
            placed = {}
            for f in sorted((repo / "snapshots").rglob("*")):
                if f.is_file():
                    out = cache / f.relative_to(download)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    blob = f.resolve()
                    if blob in placed:  # two snapshot entries share one blob
                        shutil.copyfile(placed[blob], out)
                    else:
                        shutil.move(blob, out)
                        placed[blob] = out
    finally:
        shutil.rmtree(download, ignore_errors=True)

    files = [p for p in stage_dir.rglob("*") if p.is_file()]
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files), **provenance}
