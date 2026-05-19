"""
Step 2: Build Triton Model Repository

Assembles the Triton model repository structure with NLI DeBERTa, XGBoost, and ensemble configs.

Inputs:
- workspace/nli_deberta/model.onnx: NLI DeBERTa ONNX model (with postprocessing)
- workspace/xgboost/model.onnx: XGBoost ONNX model

Outputs:
- triton-serve-ensemble/: Complete Triton model repository
  - nli_deberta/1/model.onnx: NLI model version 1
  - xgboost_classifier/1/model.onnx: XGBoost model version 1
  - ensemble_model/config.pbtxt: Ensemble orchestration config

Usage:
    python workspace/build_triton_repo.py [--workspace WORKSPACE] [--triton-repo TRITON_REPO]
"""

import argparse
import os
import shutil
from pathlib import Path

from config import get_nli_config, get_xgb_config, get_ensemble_config


def build_triton_repo(workspace: str, triton_repo: str):
    """Build Triton model repository structure."""

    if os.path.exists(triton_repo):
        print(f"  Removing existing repository at {triton_repo}")
        shutil.rmtree(triton_repo)

    # Create directory structure
    nli_ver = os.path.join(triton_repo, "nli_deberta", "1")
    xgb_ver = os.path.join(triton_repo, "xgboost_classifier", "1")
    ensemble = os.path.join(triton_repo, "ensemble_model")
    ensemble_ver = os.path.join(triton_repo, "ensemble_model", "1")

    for d in (nli_ver, xgb_ver, ensemble, ensemble_ver):
        os.makedirs(d, exist_ok=True)

    print(f"  Created directory structure:")
    print(f"    {triton_repo}/")
    print(f"    ├── nli_deberta/1/")
    print(f"    ├── xgboost_classifier/1/")
    print(f"    └── ensemble_model/1/")
    print()

    # Write Triton config files
    print(f"  Writing Triton configuration files...")
    Path(os.path.join(triton_repo, "nli_deberta", "config.pbtxt")).write_text(
        get_nli_config()
    )
    print(f"    ✓ nli_deberta/config.pbtxt")

    Path(os.path.join(triton_repo, "xgboost_classifier", "config.pbtxt")).write_text(
        get_xgb_config()
    )
    print(f"    ✓ xgboost_classifier/config.pbtxt")

    Path(os.path.join(triton_repo, "ensemble_model", "config.pbtxt")).write_text(
        get_ensemble_config()
    )
    print(f"    ✓ ensemble_model/config.pbtxt")
    print()

    # Copy NLI DeBERTa ONNX model
    print("  Copying NLI DeBERTa ONNX model...")
    nli_src = os.path.join(workspace, "nli_deberta", "model.onnx")
    if not os.path.exists(nli_src):
        raise FileNotFoundError(
            f"NLI DeBERTa ONNX model not found at {nli_src}. "
            "Run export_models.py first."
        )

    nli_dst = os.path.join(nli_ver, "model.onnx")
    shutil.copy(nli_src, nli_dst)
    onnx_size = os.path.getsize(nli_dst) / (1024 * 1024)
    print(f"    ✓ Copied model.onnx ({onnx_size:.1f} MB) to nli_deberta/1/")

    # Copy external weights if present (dynamo-based ONNX export splits weights)
    nli_data_src = nli_src + ".data"
    if os.path.exists(nli_data_src):
        nli_data_dst = os.path.join(nli_ver, "model.onnx.data")
        shutil.copy(nli_data_src, nli_data_dst)
        data_size = os.path.getsize(nli_data_dst) / (1024 * 1024)
        print(f"    ✓ Copied model.onnx.data ({data_size:.1f} MB) to nli_deberta/1/")
    print()

    # Copy XGBoost ONNX model
    print("  Copying XGBoost ONNX model...")
    xgb_src = os.path.join(workspace, "xgboost", "model.onnx")
    if not os.path.exists(xgb_src):
        raise FileNotFoundError(
            f"XGBoost ONNX model not found at {xgb_src}. "
            "Run export_models.py first."
        )

    xgb_dst = os.path.join(xgb_ver, "model.onnx")
    shutil.copy(xgb_src, xgb_dst)
    onnx_size = os.path.getsize(xgb_dst) / 1024
    print(f"    ✓ Copied model.onnx ({onnx_size:.1f} KB) to xgboost_classifier/1/")


def verify_repo_structure(triton_repo: str):
    """Verify the Triton repository structure is valid."""
    print()
    print("  Verifying repository structure...")

    required_files = [
        "nli_deberta/config.pbtxt",
        "nli_deberta/1/model.onnx",
        "xgboost_classifier/config.pbtxt",
        "xgboost_classifier/1/model.onnx",
        "ensemble_model/config.pbtxt",
    ]

    # External weights are optional (depends on ONNX export method)
    optional_files = [
        "nli_deberta/1/model.onnx.data",
    ]

    all_present = True
    for rel_path in required_files:
        full_path = os.path.join(triton_repo, rel_path)
        if os.path.exists(full_path):
            print(f"    ✓ {rel_path}")
        else:
            print(f"    ✗ {rel_path} (MISSING)")
            all_present = False

    for rel_path in optional_files:
        full_path = os.path.join(triton_repo, rel_path)
        if os.path.exists(full_path):
            print(f"    ✓ {rel_path}")

    if not all_present:
        raise RuntimeError("Repository verification failed - missing required files")

    print()
    print("  ✓ Repository structure is valid")
    print("  ✓ Both models use ONNX runtime (no Python dependencies)")


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Build Triton model repository"
    )
    parser.add_argument(
        "--workspace",
        default="workspace",
        help="Local workspace directory with model artifacts (default: workspace)",
    )
    parser.add_argument(
        "--triton-repo",
        default="triton-serve-ensemble",
        help="Output Triton model repository directory (default: triton-serve-ensemble)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Step 2: Build Triton Model Repository")
    print("=" * 70)
    print()

    build_triton_repo(args.workspace, args.triton_repo)
    verify_repo_structure(args.triton_repo)

    print()
    print("=" * 70)
    print("✓ Step 2 Complete")
    print("=" * 70)
    print()
    print(f"Triton repository ready at: {os.path.abspath(args.triton_repo)}")


if __name__ == "__main__":
    main()
