# KV-Aware Routing for Amazon SageMaker AI Endpoints

This directory contains a notebook that measures the effect of the `KV_AWARE`
routing strategy for Amazon SageMaker AI real-time endpoints.

KV-aware routing sends each request to the instance whose GPU already holds the
longest matching part of the prompt in its KV cache. The endpoint learns what
each instance has cached from KV cache events that the model container reports,
so it does not need a prefix length or a concurrency threshold.

## Notebook

[`kv_aware_routing.ipynb`](kv_aware_routing.ipynb) compares `RANDOM` and
`KV_AWARE` routing on the same model and the same requests:

- Deploys `Qwen/Qwen3-4B-Instruct-2507` on the AWS vLLM Deep Learning Container
  with KV cache events turned on
- Sets `RoutingStrategy` on the production variant (`RANDOM`, then `KV_AWARE`)
- Sends an identical multi-turn workload to each endpoint: 32 simulated users,
  4 questions each, about 12 shared documents of roughly 7,400 tokens
- Checks that both runs sent the same request bodies (a hash of the requests)
- Reads Amazon CloudWatch (`ModelLatency`, `OverheadLatency`, invocations,
  errors, GPU metrics) and client-side time to first token and cache reuse
- Prints a side-by-side table and deletes the resources it created

The notebook deploys one endpoint at a time, so it needs two GPU instances at
any moment.

## Routing Configuration

```python
"RoutingConfig": {
    "RoutingStrategy": "KV_AWARE",
}
```

The container must have prefix caching on (the vLLM default) and KV cache events
on. The notebook sets `SM_VLLM_KV_EVENTS_CONFIG` for this. Requests must use the
OpenAI-compatible Completions or Chat Completions format.

## Prerequisites

- An AWS Region where `ml.g5.xlarge` is available
- Quota for two `ml.g5.xlarge` endpoint instances
- A SageMaker execution role that can create, describe, invoke and delete
  models, endpoint configurations and endpoints, and read CloudWatch metrics
- A Jupyter environment with AWS credentials and a default Region

When you run the notebook outside SageMaker Studio, set `role` in the notebook
to a SageMaker execution role ARN. The notebook installs or upgrades `boto3`,
`pandas` and `numpy`.

## Run the Example

1. Open `kv_aware_routing.ipynb` in SageMaker Studio or another Jupyter
   environment.
2. Run the setup cells and the cell that builds the requests.
3. Run section 5 (`RANDOM`) and section 6 (`KV_AWARE`) in order. Each section
   deploys an endpoint, sends the load, reads CloudWatch and deletes the
   endpoint.
4. Run section 7 to compare the two runs.
5. Run section 8 to delete the model.

CloudWatch publishes metrics about a minute after they occur, so the notebook
waits until it can see all of a run's invocations.

## Reading the Results

The results depend on the workload. The example workload reuses long documents
across several turns, which is the case KV-aware routing is built for. To
measure your own traffic, change the `workload` settings or replace the requests
the notebook builds. Run the comparison more than once, with a new `doc_seed`
each time, before you conclude that a difference is larger than run-to-run
variation.

## Cost

The notebook creates SageMaker AI endpoints that bill while they run. Run the
cleanup cells at the end, and check the SageMaker console for endpoints left
behind if a run stops early.

## License

This sample is released under the license in the repository root.
