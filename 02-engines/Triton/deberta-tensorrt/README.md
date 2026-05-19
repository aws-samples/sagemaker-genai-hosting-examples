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

## Production note: JIT vs. ahead-of-time compilation

This example ships a `model.onnx` artifact and lets ONNX Runtime JIT-compile the TensorRT engine on the endpoint at model load time. That keeps the artifact portable across GPU architectures (T4, A10G, L4, H100, …) but adds a **cold-start penalty** on the first load as TRT builds the engine.

For production, pre-compile a `model.plan` with `trtexec` on the target GPU family (TRT engines are locked to a specific compute capability and TRT version) and switch the config to `platform: "tensorrt_plan"`. You trade GPU portability for fast, deterministic endpoint startup.

### How to integrate a pre-compiled `.plan` into this flow

Reference: [NVIDIA TensorRT — Deploy to Triton quickstart](https://github.com/NVIDIA/TensorRT/tree/main/quickstart/deploy_to_triton).

1. **Build the engine on a host with the target GPU arch** (e.g. a `g5.2xlarge` for A10G / SM 8.6). Easiest path is to run `trtexec` inside the same SageMaker Triton container image the endpoint will use, so TRT versions match exactly. After running `python workspace/export_model.py` to produce `workspace/nli_deberta/model.onnx`:

   ```bash
   trtexec \
     --onnx=workspace/nli_deberta/model.onnx \
     --saveEngine=workspace/nli_deberta/model.plan \
     --fp16 \
     --minShapes=input_ids:1x128,attention_mask:1x128 \
     --optShapes=input_ids:50x128,attention_mask:50x128 \
     --maxShapes=input_ids:50x128,attention_mask:50x128 \
     --memPoolSize=workspace:1024
   ```

2. **Swap the artifact** in the Triton model repo that the notebook builds in section 3 — copy `model.plan` into `triton-serve/nli_deberta/1/` instead of `model.onnx` (+ `model.onnx.data`).

3. **Replace `workspace/config.pbtxt`** — drop the `backend: "onnxruntime"` + `optimization { execution_accelerators ... }` block and use the TRT backend directly:

   ```
   name: "nli_deberta"
   platform: "tensorrt_plan"
   max_batch_size: 0

   input [
     { name: "input_ids"      data_type: TYPE_INT64  dims: [ 50, 128 ] },
     { name: "attention_mask" data_type: TYPE_INT64  dims: [ 50, 128 ] }
   ]
   output [
     { name: "confidence_vector" data_type: TYPE_FP32 dims: [ 1, 50 ] }
   ]
   instance_group [ { kind: KIND_GPU, count: 1 } ]
   ```

4. **Keep `INSTANCE_TYPE` pinned** to the same GPU family you compiled for. Crossing architectures (e.g. g5 → g6) requires rebuilding the `.plan`.

The rest of the notebook (tar, S3 upload, `ModelBuilder`, invocation, cleanup) is unchanged.

## Files

```
deberta-tensorrt/
├── deberta_triton_tensorrt.ipynb   # End-to-end deployment notebook
└── workspace/
    ├── config.pbtxt                # Triton model config (TRT FP16 block)
    └── export_model.py             # ONNX export with baked-in postprocessing
```

Start with the notebook.
