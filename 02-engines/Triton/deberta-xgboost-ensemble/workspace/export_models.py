"""
Step 1: Prepare Models

Exports NLI DeBERTa model (with postprocessing) and trains/exports XGBoost classifier to ONNX format.

Architecture:
  NLI DeBERTa: (N, seq_len) → logits (N, 3) → 2-class softmax → entailment probs → (1, N)
  XGBoost:     (1, N) confidence vector → binary prediction

Both models are exported to ONNX for use with Triton's onnxruntime backend.

Outputs:
- workspace/nli_deberta/: NLI DeBERTa ONNX model with baked-in postprocessing
- workspace/xgboost/: XGBoost ONNX model

Usage:
    python workspace/export_models.py [--workspace WORKSPACE]
    python workspace/export_models.py --step nli        # export NLI model only
    python workspace/export_models.py --step xgboost    # train XGBoost only
"""

import argparse
import os


def export_nli_to_onnx(workspace: str, max_seq_len: int = 128, n_labels: int = 50):
    """Export NLI DeBERTa model to ONNX with baked-in postprocessing.

    The exported model takes N tokenized (premise, hypothesis) pairs and outputs
    a (1, N) confidence vector of entailment probabilities — ready for XGBoost.

    Postprocessing matches HuggingFace ZeroShotClassificationPipeline with
    multi_label=True: 2-class softmax over [contradiction, entailment], then
    extract the entailment probability.
    """
    import torch
    import torch.nn as nn
    from transformers import AutoModelForSequenceClassification

    from config import NLI_MODEL_ID, ENTAILMENT_IDX, CONTRADICTION_IDX

    dest = os.path.join(workspace, "nli_deberta")
    os.makedirs(dest, exist_ok=True)

    print(f"  Downloading {NLI_MODEL_ID}...")
    base_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID)
    base_model.eval()

    class NLIWithPostprocess(nn.Module):
        """NLI model with baked-in postprocessing for ONNX export.

        Takes N (premise, hypothesis) pairs and returns a (1, N) vector of
        entailment probabilities.
        """

        def __init__(self, model, entailment_idx=1, contradiction_idx=0):
            super().__init__()
            self.model = model
            self.model.config.return_dict = False  # tuple output for clean ONNX export
            self.entailment_idx = entailment_idx
            self.contradiction_idx = contradiction_idx

        def forward(self, input_ids, attention_mask):
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output[0]  # (N, 3)

            # 2-class softmax over [contradiction, entailment] — matches
            # HuggingFace ZeroShotClassificationPipeline multi_label=True
            entail_contr = logits[:, [self.contradiction_idx, self.entailment_idx]]  # (N, 2)
            probs = torch.softmax(entail_contr, dim=-1)  # (N, 2)
            confidence = probs[:, 1]  # (N,) — entailment probability

            return confidence.unsqueeze(0)  # (1, N)

    wrapped_model = NLIWithPostprocess(base_model, ENTAILMENT_IDX, CONTRADICTION_IDX)
    wrapped_model.eval()

    print(f"  Exporting to ONNX (n_labels={n_labels}, max_seq_len={max_seq_len})...")

    dummy_input_ids = torch.zeros((n_labels, max_seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones((n_labels, max_seq_len), dtype=torch.long)

    onnx_path = os.path.join(dest, "model.onnx")
    # dynamo=True: the legacy TorchScript exporter writes a single protobuf and
    # blows past the 2GB cap for DeBERTa-v3 (disentangled attention + per-layer
    # relative position embeddings serialize to ~2.2GB). The dynamo exporter
    # writes weights to an external `.data` sidecar automatically, which Triton's
    # ONNX backend reads transparently.
    torch.onnx.export(
        wrapped_model,
        (dummy_input_ids, dummy_attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["confidence_vector"],
        opset_version=18,
        dynamo=True,
        external_data=True,
    )

    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    data_path = onnx_path + ".data"
    data_msg = ""
    if os.path.exists(data_path):
        data_size_mb = os.path.getsize(data_path) / (1024 * 1024)
        data_msg = f"\n    - model.onnx.data ({data_size_mb:.1f} MB)"
    else:
        raise RuntimeError(
            f"Expected external weights file at {data_path}. "
            "Dynamo exporter should have produced it. "
            "Check torch version (>=2.5 required for dynamo=True)."
        )

    print(f"  ✓ NLI DeBERTa ONNX model saved to {dest}")
    print(f"    - model.onnx ({onnx_size_mb:.1f} MB){data_msg}")
    print(f"    - Includes postprocessing (softmax + entailment extraction)")
    print(f"    - Input: input_ids, attention_mask [{n_labels}, {max_seq_len}]")
    print(f"    - Output: confidence_vector [1, {n_labels}]")


def train_demo_xgboost(n_labels: int = 50):
    """Train a demo XGBoost classifier on *synthetic* data.

    ⚠ The returned classifier has no real-world skill — its training data is
    generated with sklearn.make_classification, which has nothing to do with
    the NLI confidence vectors it sees at inference time. This exists purely
    to produce a working end-to-end pipeline.

    To adapt this example to your own workload, replace this function with
    your own classifier (or skip it entirely and pass a pre-trained XGBClassifier
    to convert_xgboost_to_onnx() below).
    """
    import numpy as np
    import xgboost as xgb
    from sklearn.datasets import make_classification

    print(f"  Generating synthetic {n_labels}-dim training data (NOT real NLI scores)...")
    X_train, y_train = make_classification(
        n_samples=1000,
        n_features=n_labels,
        n_informative=20,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )
    # Clip to [0, 1] to approximate confidence score distribution
    X_train = np.clip(X_train, 0, 1).astype(np.float32)

    print("  Training XGBoost classifier...")
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="binary:logistic",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def convert_xgboost_to_onnx(clf, workspace: str, n_labels: int = 50):
    """Convert any trained XGBoost classifier to ONNX and save under workspace/xgboost/.

    Drop your own trained XGBClassifier in here — the ONNX contract (input name
    'float_input', shape [1, n_labels], outputs 'label' + 'probabilities') is
    what the Triton config.pbtxt for xgboost_classifier expects. If you change
    the input dimension, update N_LABELS in config.py to match.
    """
    import onnx
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    dest = os.path.join(workspace, "xgboost")
    os.makedirs(dest, exist_ok=True)

    print("  Converting XGBoost model to ONNX...")
    initial_type = [("float_input", FloatTensorType([1, n_labels]))]
    onnx_model = convert_xgboost(clf, initial_types=initial_type, target_opset=12)

    onnx_path = os.path.join(dest, "model.onnx")
    onnx.save_model(onnx_model, onnx_path)

    # Validate the ONNX output names match what the Triton config expects.
    # onnxmltools occasionally renames outputs across versions; surfacing this
    # here is much easier to debug than a Triton load failure.
    loaded_model = onnx.load(onnx_path)
    output_names = [output.name for output in loaded_model.graph.output]
    expected_outputs = ["label", "probabilities"]

    if set(output_names) != set(expected_outputs):
        print(f"  ⚠ WARNING: ONNX outputs {output_names} don't match expected {expected_outputs}")
        print(f"  Update config.py if this changes in future onnxmltools versions")
    else:
        print(f"  ✓ ONNX outputs validated: {output_names}")

    print(f"  ✓ XGBoost ONNX model saved to {dest}")
    print(f"    - model.onnx ({os.path.getsize(onnx_path) / 1024:.1f} KB)")
    print(f"    - Input: float_input [1, {n_labels}]")


def train_and_export_xgboost(workspace: str, n_labels: int = 50):
    """Demo convenience: train on synthetic data and export. Replace both steps
    with your own classifier for a real workload — see train_demo_xgboost().
    """
    clf = train_demo_xgboost(n_labels=n_labels)
    convert_xgboost_to_onnx(clf, workspace=workspace, n_labels=n_labels)


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Export NLI DeBERTa and train XGBoost to ONNX"
    )
    parser.add_argument(
        "--workspace",
        default="workspace",
        help="Local workspace directory for model artifacts (default: workspace)",
    )
    parser.add_argument(
        "--step",
        choices=["nli", "xgboost"],
        default=None,
        help="Run a single step instead of both (avoids process state conflicts)",
    )
    args = parser.parse_args()

    from config import N_LABELS, MAX_SEQ_LEN

    print("=" * 70)
    print("Step 1: Prepare Models (ONNX)")
    print("=" * 70)
    print()

    os.makedirs(args.workspace, exist_ok=True)

    if args.step is None or args.step == "nli":
        print("[1/2] Exporting NLI DeBERTa to ONNX...")
        export_nli_to_onnx(args.workspace, MAX_SEQ_LEN, N_LABELS)
        print()

    if args.step is None or args.step == "xgboost":
        print("[2/2] Training and exporting XGBoost to ONNX...")
        train_and_export_xgboost(args.workspace, N_LABELS)
        print()

    print("=" * 70)
    print("✓ Step 1 Complete")
    print("=" * 70)
    print()
    print(f"Models saved to: {os.path.abspath(args.workspace)}")
    print("  - nli_deberta/model.onnx (NLI DeBERTa with postprocessing)")
    print("  - xgboost/model.onnx (XGBoost classifier)")
    print()
    print("Both models use ONNX format - no Python dependencies needed in Triton!")


if __name__ == "__main__":
    main()
