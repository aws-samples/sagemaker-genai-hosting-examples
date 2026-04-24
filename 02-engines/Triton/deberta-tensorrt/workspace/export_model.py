"""Export NLI DeBERTa to ONNX with baked-in postprocessing.

The exported model takes 50 tokenized (premise, hypothesis) pairs and outputs
a (1, 50) confidence vector of entailment probabilities. Postprocessing
(2-class softmax + entailment extraction) is part of the ONNX graph, so
Triton runs pure C++ at inference time with no Python glue.

Usage:
    python workspace/export_model.py
"""

import os

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

MODEL_ID = "cross-encoder/nli-deberta-v3-base"
N_LABELS = 50
MAX_SEQ_LEN = 128

# config.json: {0: "contradiction", 1: "entailment", 2: "neutral"}
CONTRADICTION_IDX = 0
ENTAILMENT_IDX = 1


class NLIWithPostprocess(nn.Module):
    """NLI model + baked-in postprocessing.

    Matches HuggingFace ZeroShotClassificationPipeline with multi_label=True:
    2-class softmax over [contradiction, entailment], return entailment probs.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.config.return_dict = False

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        entail_contr = logits[:, [CONTRADICTION_IDX, ENTAILMENT_IDX]]
        probs = torch.softmax(entail_contr, dim=-1)
        return probs[:, 1].unsqueeze(0)


def main():
    dest = "workspace/nli_deberta"
    os.makedirs(dest, exist_ok=True)

    print(f"Downloading {MODEL_ID}...")
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    base_model.eval()

    wrapped = NLIWithPostprocess(base_model)
    wrapped.eval()

    print(f"Exporting to ONNX (n_labels={N_LABELS}, max_seq_len={MAX_SEQ_LEN})...")
    dummy_input_ids = torch.zeros((N_LABELS, MAX_SEQ_LEN), dtype=torch.long)
    dummy_attention_mask = torch.ones((N_LABELS, MAX_SEQ_LEN), dtype=torch.long)

    onnx_path = os.path.join(dest, "model.onnx")
    torch.onnx.export(
        wrapped,
        (dummy_input_ids, dummy_attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["confidence_vector"],
        opset_version=18,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    extras = ""
    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        extras = f" + model.onnx.data ({os.path.getsize(data_path) / (1024 * 1024):.1f} MB)"
    print(f"Saved {onnx_path} ({size_mb:.1f} MB){extras}")


if __name__ == "__main__":
    main()
