# SageMaker HyperPod Inference

Use HyperPod Inference when the deployment requires Kubernetes-native control
over distributed GPU infrastructure.

## Select the runtime

- Choose the sample's JumpStart path for an AWS-managed container and model
  configuration.
- Choose a custom runtime only when the workload requires a specific serving
  engine, parser, quantisation option, argument, or parallel layout.

For a custom runtime, keep these values compatible as one unit:

- checkpoint variant and immutable model revision;
- instance type, GPU count, node count, replicas, and parallel layout;
- runtime image pinned by digest;
- parser and security configuration;
- context length, batching, KV cache, and memory settings.

Do not guess minimum runtime versions, image digests, parser names, or model
commits. Read them from the current sample.

## Configure and validate

1. Copy the sample's ignored configuration examples.
2. Keep credentials out of configuration files and manifests.
3. Display the Kubernetes context, API server, namespace, resource name,
   instance, GPU count, model source, and image before requesting approval.
4. Run local validation.
5. Use the sample's server dry run when the cluster is available.

When creating or rotating a Secret, require exact confirmation of the context,
API server, namespace, and Secret name. Do not overwrite an existing Secret
unless rotation was explicitly requested.

## Deploy, test, and clean up

Use the sample's bounded deployment command rather than applying templates
directly. Wait for a terminal custom-resource state and capture diagnostics
without exposing credentials or request bodies.

Run one bounded smoke test. Prefer a temporary local port-forward unless the
user supplied an authenticated route. Require HTTPS for authenticated remote
endpoints.

Use exact-name guards during cleanup. Wait until the namespaced resource is
absent, then inventory worker nodes, load balancers, storage, caches, images,
model copies, and reserved capacity that remain.

## Optional production capabilities

Discuss these only when they solve a stated requirement:

- model and image caching;
- Amazon S3 or Amazon FSx checkpoint sources;
- Inference Gateway and endpoint routing;
- KV caching and intelligent routing;
- replicas, autoscaling, and rollout strategy.

Do not report a capability as tested unless it was enabled and observed in the
current environment.
