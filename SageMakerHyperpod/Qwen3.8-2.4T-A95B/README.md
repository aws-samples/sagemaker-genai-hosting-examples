# Qwen3.8-2.4T-A95B

Deployment config and a minimal client for serving **Qwen3.8-2.4T-A95B** (NVFP4 quantized) with [vLLM](https://github.com/vllm-project/vllm) behind an OpenAI-compatible API.

## Contents

| File | Purpose |
|------|---------|
| `qwen.yaml` | SageMaker `InferenceEndpointConfig` that serves the model with vLLM |
| `qwen_client.py` | Zero-dependency CLI client for the endpoint (stdlib `urllib` only) |

## Deployment

`qwen.yaml` defines an `InferenceEndpointConfig` named `qwen38`:

- **Model:** `Inferact/Qwen3.8-2.4T-A95B-NVFP4` pulled from Hugging Face
- **Instance:** `ml.p6-b300.48xlarge`, 8× GPU, tensor-parallel size 8
- **Served as:** `Qwen3.8` on container port `8000` (`v1/chat/completions`)
- **vLLM flags:** prefix caching, auto tool choice (`qwen3_coder` parser), reasoning parser (`qwen3`), `flashinfer_cutedsl` linear backend

Apply it to the cluster:

```bash
kubectl apply -f qwen.yaml
```

## Using the client

The endpoint is not exposed externally, so port-forward to it first:

```bash
kubectl port-forward pod/qwen38-7fbb468f4c-2cw44 8000:8000 &
```

Then invoke `qwen_client.py`. It has no dependencies beyond the Python standard library.

```bash
# One-shot prompt
python qwen_client.py "Explain tensor parallelism in one paragraph."

# Stream the response as it's generated
python qwen_client.py --stream "Write a haiku about GPUs."

# Read the prompt from stdin
echo "What are you?" | python qwen_client.py -
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--base-url` | `http://localhost:8000` | Endpoint base URL |
| `--model` | `Qwen3.8` | Served model name |
| `--api-key` | _(none)_ | Bearer token, if the endpoint requires one |
| `--system` | _(none)_ | Optional system prompt |
| `--max-tokens` | `1024` | Max tokens to generate |
| `--temperature` | `0.7` | Sampling temperature |
| `--stream` | off | Stream tokens as they arrive |
| `--show-reasoning` | off | Also print the model's reasoning / chain-of-thought |

Token usage (`prompt` / `completion` / `total`) is printed to stderr after non-streaming calls. If the connection fails, the client reminds you to check the port-forward.

## Requirements

- Python 3.7+ (uses `from __future__ import annotations`; standard library only)
- `kubectl` access to the cluster running the endpoint
