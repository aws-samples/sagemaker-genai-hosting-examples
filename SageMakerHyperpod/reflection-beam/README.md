# Reflection Beam on SageMaker HyperPod Inference

This sample provides three Reflection Beam deployment paths for Amazon
SageMaker HyperPod Inference:

- SageMaker JumpStart with `JumpStartModel`.
- SGLang with `InferenceEndpointConfig`.
- vLLM with `InferenceEndpointConfig`.

The sample includes the public JumpStart model IDs. The SGLang path loads a
selected Beam BF16, FP8, or NVFP4 target checkpoint and configures DFlash2 as a
separate speculative drafter. The vLLM path serves the selected target
checkpoint without speculative decoding until its released Beam configuration
is confirmed. Supply public image digests built from the listed SGLang and vLLM
release commits. Do not add credentials, account resources, or model request
bodies to the committed configuration.

## Files

| Path | Purpose |
|---|---|
| `reflection-beam-jumpstart.yaml` | JumpStart manifest template |
| `reflection-beam-sglang.yaml` | SGLang manifest template |
| `reflection-beam-vllm.yaml` | vLLM manifest template |
| `config/deployment.env.example` | Cluster and topology settings |
| `config/release.env.example` | Isolated public release identifiers |
| `hyperpod_tool.py` | Validate, deploy, inspect, invoke, smoke test, and clean up |
| `beam_client.py` | OpenAI-compatible streaming client and guarded review example |
| `example/pagination.py` | Fictitious input for the guarded review example |
| `tests/` | Standard-library unit tests |

The YAML files are templates. Run them through `hyperpod_tool.py`; do not apply
them directly.

## Prerequisites

- An EKS-orchestrated SageMaker HyperPod cluster configured for model
  deployment.
- `kubectl` access to the intended cluster.
- Python 3.11 or later.
- GPU capacity and topology supported by the public checkpoint and runtime.
- The AWS CLI for JumpStart endpoint invocation.
- The `openai` package for SGLang or vLLM invocation.

Install the client dependency in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Configure

Create ignored local configuration files:

```bash
cp config/deployment.env.example config/deployment.env
cp config/release.env.example config/release.env
```

Set the deployment values for the path you will use. Keep runtime-specific
values together; a checkpoint, image, parser, quantization mode, and parallel
layout must be compatible with each other.

For SGLang, set `SGLANG_MODEL_ID` and `SGLANG_MODEL_COMMIT_SHA` to the target
Beam checkpoint. The example release configuration selects NVFP4 and requires
you to add its full public commit SHA. The manifest loads
`reflection/Beam-1.0-501B-A23B-DFlash2` separately with the DFlash algorithm,
unified memory, and a draft block size of 8.

Set the SGLang GPU count and tensor parallel size for the selected target and
instance type. The supplied launch guidance uses TP4 for NVFP4 or FP8 on B300,
TP8 for BF16 on B300, and TP8 for FP8 or BF16 on H200. Keep the page size,
context length, memory fraction, attention backend, target checkpoint, and
draft checkpoint compatible with the released runtime.

The vLLM template does not enable DFlash2. Its example configuration uses the
NVFP4 checkpoint with the AWS vLLM DLC pinned by image digest.

Use these runtime support commits:

| Runtime | Minimum version | Release commit |
|---|---|---|
| SGLang | `0.5.23` (provisional, pending public release) | [`8f94384cb5caeafc57dfa23c52bcfcedf6b88a41`](https://github.com/radixark/sglang-reflection-oss/commit/8f94384cb5caeafc57dfa23c52bcfcedf6b88a41) |
| vLLM | AWS vLLM DLC `2.7` | [`8e0922e88bf9d4d5c0031f817a4e4749c9ca4ec2`](https://github.com/Inferact/lin/commit/8e0922e88bf9d4d5c0031f817a4e4749c9ca4ec2) |

The source commit is not a deployable image reference. The example pins the
public AWS vLLM DLC by digest. Set `SGLANG_IMAGE` to the public SGLang image
built with Beam support and pin it by digest.

For the JumpStart path, `config/release.env.example` defaults to the NVFP4
checkpoint and lists the BF16 and FP8 model IDs as alternatives. Select the
model ID and instance type supported by your chosen checkpoint variant. Keep
`JUMPSTART_MODEL_VERSION=1.0.0` for a reproducible deployment, or set it to
`latest` to omit `modelVersion` from the rendered manifest and resolve the
latest available version when you deploy.

The tool reads non-executable `KEY=VALUE` files. It rejects shell `export`
syntax, credential fields, blank values, example placeholders, unpinned custom
runtime images, short model commits, invalid namespaces, and non-positive GPU
or parallelism values. Environment variables override file values.

Validate locally before touching the cluster:

```bash
python hyperpod_tool.py \
  --config config/deployment.env \
  --config config/release.env \
  validate sglang
```

Replace `sglang` with `jumpstart` or `vllm`. Add `--server-dry-run` to ask the
current cluster to validate the rendered resource without creating it:

```bash
python hyperpod_tool.py \
  --config config/deployment.env \
  --config config/release.env \
  validate vllm \
  --server-dry-run
```

## Deploy

Deploy one path and wait for a terminal deployment state:

```bash
python hyperpod_tool.py \
  --config config/deployment.env \
  --config config/release.env \
  deploy sglang \
  --timeout-seconds 1800
```

The command applies the rendered manifest through standard input, so it does
not write a populated manifest to disk. It polls with a bounded timeout and
prints resource and pod diagnostics when deployment fails or times out.

Use `--context` or `--kubeconfig` before the subcommand when the current
`kubectl` target is not explicit:

```bash
python hyperpod_tool.py \
  --context my-hyperpod-context \
  --config config/deployment.env \
  --config config/release.env \
  deploy jumpstart
```

## Status

Read current status without waiting:

```bash
python hyperpod_tool.py \
  --config config/deployment.env \
  status sglang
```

Wait for deployment completion with a fixed deadline:

```bash
python hyperpod_tool.py \
  --config config/deployment.env \
  status sglang \
  --wait \
  --timeout-seconds 1800
```

Use `--output json` for automation. Status and cleanup only need the deployment
configuration because they target fixed sample resource names.

## Invoke SGLang or vLLM

Invoke an OpenAI-compatible endpoint you can already reach:

```bash
python hyperpod_tool.py invoke sglang \
  --base-url http://127.0.0.1:8000/v1 \
  --prompt "Explain tensor parallelism in two sentences." \
  --temperature 1.0 \
  --top-p 0.7 \
  --request-timeout-seconds 120
```

Reflection recommends `temperature=1.0` and `top_p=0.7` as starting sampling
settings for Beam. Tune them for your workload after confirming output quality.

For an authenticated gateway, pass a short-lived token through the environment:

```bash
read -rsp "Beam API token: " BEAM_API_KEY && export BEAM_API_KEY
echo
python hyperpod_tool.py invoke vllm \
  --base-url https://beam.example.com/v1 \
  --prompt "Reply with one sentence."
unset BEAM_API_KEY
```

Do not place credentials in a URL, config file, manifest, or command argument.

## Smoke test SGLang or vLLM

Run a bounded private smoke test:

```bash
python hyperpod_tool.py \
  --config config/deployment.env \
  smoke sglang
```

The command:

1. Confirms that the custom resource reports `DeploymentComplete`.
2. Reads the worker selector from resource status.
3. Selects a running, ready pod deterministically.
4. Starts a temporary port-forward on an available local port.
5. Checks `/health` and sends a short OpenAI-compatible request.
6. Stops the port-forward on success, failure, timeout, or interruption.

Pass `--base-url` to test an existing route without starting a port-forward.

## Invoke or smoke test JumpStart

The JumpStart request schema depends on the released model package. Create a
JSON request body that matches its public documentation. Store it in
`config/jumpstart-request.json`; the sample ignores local JSON request files.

Invoke the SageMaker endpoint:

```bash
python hyperpod_tool.py \
  --config config/deployment.env \
  invoke jumpstart \
  --request-file config/jumpstart-request.json \
  --region us-west-2
```

Check the HyperPod resource state before invoking:

```bash
python hyperpod_tool.py \
  --config config/deployment.env \
  smoke jumpstart \
  --request-file config/jumpstart-request.json \
  --region us-west-2
```

The tool passes the body to `aws sagemaker-runtime invoke-endpoint` without
assuming a Beam release payload or response schema.

## Guarded code-review example

Run the read-only tool-use example against SGLang or vLLM:

```bash
python beam_client.py review \
  --base-url http://127.0.0.1:8000/v1 \
  --workspace-root example \
  --target pagination.py \
  --timeout-seconds 120
```

Review mode allows exactly one read of the approved target inside the supplied
workspace root. It rejects path escapes, unsupported file types, and oversized
files. It prints a proposed unified diff for human review and never writes or
applies the patch.

## Clean up

Cleanup requires the exact resource name as a guard against deleting the wrong
deployment. The following example removes the JumpStart resource:

```bash
python hyperpod_tool.py \
  --config config/deployment.env \
  cleanup jumpstart \
  --confirm-name reflection-beam-jumpstart \
  --timeout-seconds 600
```

Use the matching runtime and exact confirmation:

| Runtime | Confirmation |
|---|---|
| JumpStart | `reflection-beam-jumpstart` |
| SGLang | `reflection-beam-sglang` |
| vLLM | `reflection-beam-vllm` |

The command deletes only that namespaced custom resource and waits for bounded
confirmation that it is absent.

## Run tests

```bash
python -m unittest discover -s tests -v
```

The tests do not require a cluster, AWS credentials, or the `openai` package.

## References

- [SageMaker HyperPod model deployment](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-model-deployment.html)
- [Deploy JumpStart models](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-model-deployment-deploy.html)
- [Deploy custom models](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-model-deployment-deploy-ftm.html)
- [Reflection Beam model collection](https://huggingface.co/collections/reflection/beam)
- [Beam DFlash2 draft checkpoint](https://huggingface.co/reflection/Beam-1.0-501B-A23B-DFlash2/tree/8d26c952ec6a82a9a18825d6126c9f6948149ad9)
- [SGLang Beam release commit](https://github.com/radixark/sglang-reflection-oss/commit/8f94384cb5caeafc57dfa23c52bcfcedf6b88a41)
- [vLLM Beam release commit](https://github.com/Inferact/lin/commit/8e0922e88bf9d4d5c0031f817a4e4749c9ca4ec2)
