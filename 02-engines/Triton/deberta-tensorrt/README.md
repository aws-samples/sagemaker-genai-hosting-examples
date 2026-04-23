# DeBERTa on Triton with TensorRT FP16

Deploy [`cross-encoder/nli-deberta-v3-base`](https://huggingface.co/cross-encoder/nli-deberta-v3-base) to Amazon SageMaker on NVIDIA Triton, accelerated with ONNX Runtime's TensorRT FP16 execution provider.

A single notebook walks through ONNX export → Triton model repository → SageMaker Inference Component → sample inference.

## Why Triton for this model

A base HuggingFace / PyTorch container runs the full Python stack on every request. Triton's inference hot path is pure C++, which unlocks the lower-level GPU optimisations (TensorRT, FP16, graph fusion) that cannot be applied to a live Python model.

## What's in the config

The full optimisation block lives in [`workspace/config.pbtxt`](workspace/config.pbtxt):

```
optimization {
  graph { level: 3 }
  execution_accelerators {
    gpu_execution_accelerator [
      { name: "tensorrt"
        parameters { key: "precision_mode" value: "FP16" } }
    ]
  }
}
```

- `graph.level: 3` — ONNX Runtime operator fusion (e.g. `MatMul + Add + Gelu → FusedMatMul`), constant folding, redundant-transpose elimination.
- TensorRT FP16 — replaces generic ORT kernels with TensorRT; JIT-compiles layer fusion and kernel plans for the specific GPU SKU.

## Files

```
deberta-tensorrt/
├── deberta_triton_tensorrt.ipynb   # End-to-end deployment notebook
└── workspace/
    ├── config.pbtxt                # Triton model config (TRT FP16 block)
    └── export_model.py             # ONNX export with baked-in postprocessing
```

Start with the notebook.
