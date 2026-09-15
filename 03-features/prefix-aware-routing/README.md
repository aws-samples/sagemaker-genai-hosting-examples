# Prefix-Aware Routing for Amazon SageMaker AI Endpoints

This directory contains a notebook demonstrating SageMaker AI's new
`PREFIX_AWARE` routing strategy for real-time inference endpoints.

Prefix-aware routing examines the beginning of each request and consistently
routes requests with the same prefix to the same endpoint instance. This
improves KV-cache reuse for workloads with repeated system prompts, shared
documents, or other common prompt prefixes.

## Notebook

[`prefix_aware_routing.ipynb`](prefix_aware_routing.ipynb) demonstrates how to:

- Deploy `Qwen/Qwen3.5-4B` with the SageMaker vLLM 0.26.0 container
- Configure prefix-aware routing on a production variant
- Use either a single-model endpoint or an Inference Component
- Invoke the endpoint through the native SageMaker Runtime API
- Invoke a single-model endpoint through the OpenAI-compatible API
- Run repeated requests to observe latency and KV-cache reuse
- Delete the resources created by the example

The example uses two `ml.g6e.4xlarge` instances so that requests can be routed
across multiple model-serving instances.

## Routing Configuration

The notebook adds `RoutingConfig` to the production variant:

```python
"RoutingConfig": {
    "RoutingStrategy": "PREFIX_AWARE",
    "PrefixAwareRoutingConfig": {
        "PrefixLength": 1024,
        "ConcurrencyThreshold": 10,
    },
}
```

The two prefix-aware routing parameters are:

| Parameter | Valid range | Description |
| --- | ---: | --- |
| `PrefixLength` | 1,024-65,536 | Amount of each request used to select a target instance. This is measured in bytes for the native SageMaker Invoke API and in characters from the extracted message text for the OpenAI-compatible API. |
| `ConcurrencyThreshold` | 1-1,024 | Maximum number of in-flight requests allowed on the preferred instance before SageMaker routes overflow traffic to a less busy instance. |

Set `PrefixLength` high enough to include the shared prompt prefix while also
including enough request-specific content to distribute unrelated workloads.
Tune `ConcurrencyThreshold` to balance cache locality with overload protection.

## Deployment Options

The notebook contains two alternative deployment paths:

1. **Single-model endpoint (SME):** The model is specified directly in the
   production variant. This path includes native and OpenAI-compatible
   invocation examples.
2. **Inference Component (IC):** The endpoint is created first, and two copies
   of an Inference Component are attached with one accelerator requested per
   copy.

Run only one deployment path in a notebook session. Both paths reuse the same
generated resource names.

## Prerequisites

- An AWS Region that supports the selected SageMaker instance and
  `PREFIX_AWARE` routing
- A SageMaker execution role with permissions to create, describe, invoke, and
  delete models, endpoint configurations, endpoints, and Inference Components
- Quota for two `ml.g6e.4xlarge` endpoint instances
- A Jupyter environment with AWS credentials and a configured default Region

When running outside SageMaker Studio, set `role` in the notebook to a
SageMaker execution role ARN. The notebook installs or upgrades `boto3` and
installs the `openai` package for the OpenAI-compatible invocation example.

## Run the Example

1. Open `prefix_aware_routing.ipynb` in SageMaker Studio or another Jupyter
   environment configured for AWS.
2. Run the setup and model configuration cells.
3. Choose and run either the **SME endpoint** or **IC endpoint** deployment
   cells.
4. Run the corresponding inference cells.
5. Run the repeated-prefix performance test and inspect the model container's
   CloudWatch logs for prompt-throughput and cache-reuse behavior.
6. Run the cleanup cells to avoid continued endpoint charges.

## Expected Impact

The notebook reports benchmark observations for Llama 3.1 70B in which
prefix-aware routing reduced median time to first token by up to 77%, increased
KV-cache hit rates from approximately 25% to more than 80%, and increased
throughput by up to 16%.

These figures are workload-specific. Results depend on prompt-prefix reuse,
request concurrency, model configuration, cache capacity, and the selected
`PrefixLength` and `ConcurrencyThreshold` values.
