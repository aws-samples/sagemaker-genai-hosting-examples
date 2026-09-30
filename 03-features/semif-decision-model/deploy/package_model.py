#!/usr/bin/env python3
"""Create model.tar.gz from pinned GGUF and tokenizer revisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

GGUF_REPO = "bartowski/Qwen_Qwen3.5-4B-GGUF"
GGUF_REVISION = "4168f45a16a1290d65a4ec0fa312ae917a4c15d6"
GGUF_FILE = "Qwen_Qwen3.5-4B-Q4_K_M.gguf"
GGUF_SHA256 = "13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983"
TOKENIZER_REPO = "Qwen/Qwen3.5-4B"
TOKENIZER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TOKENIZER_PATTERNS = ("*.json", "*.jinja", "*.model", "*.txt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_bundle(output: Path, cache_dir: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    from huggingface_hub import hf_hub_download, snapshot_download

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="semif-model-") as temporary:
        root = Path(temporary)
        gguf = Path(
            hf_hub_download(
                repo_id=GGUF_REPO,
                revision=GGUF_REVISION,
                filename=GGUF_FILE,
                cache_dir=cache_dir,
            )
        )
        gguf_sha256 = sha256(gguf)
        if gguf_sha256 != GGUF_SHA256:
            raise ValueError(
                "GGUF SHA-256 mismatch: "
                f"expected {GGUF_SHA256}, received {gguf_sha256}"
            )
        tokenizer_snapshot = Path(
            snapshot_download(
                repo_id=TOKENIZER_REPO,
                revision=TOKENIZER_REVISION,
                allow_patterns=list(TOKENIZER_PATTERNS),
                cache_dir=cache_dir,
            )
        )
        shutil.copy2(gguf, root / GGUF_FILE)
        tokenizer_target = root / "tokenizer"
        tokenizer_target.mkdir()
        for source in sorted(tokenizer_snapshot.rglob("*")):
            if source.is_file():
                relative = source.relative_to(tokenizer_snapshot)
                target = tokenizer_target / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        manifest = {
            "gguf": {
                "repo": GGUF_REPO,
                "revision": GGUF_REVISION,
                "file": GGUF_FILE,
                "sha256": gguf_sha256,
            },
            "tokenizer": {
                "repo": TOKENIZER_REPO,
                "revision": TOKENIZER_REVISION,
            },
        }
        (root / "model-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        with tarfile.open(output, "w:gz") as archive:
            for path in sorted(root.rglob("*")):
                archive.add(path, arcname=path.relative_to(root))
    return {**manifest, "archive": str(output), "archive_sha256": sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(create_bundle(args.output, args.cache_dir), indent=2))


if __name__ == "__main__":
    main()
