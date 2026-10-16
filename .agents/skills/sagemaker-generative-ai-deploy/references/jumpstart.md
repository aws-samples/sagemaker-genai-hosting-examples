# SageMaker JumpStart

## Choose the JumpStart path

Use a JumpStart real-time endpoint when the user wants a managed model package,
HTTPS endpoint, and invocation API without operating Kubernetes.

Use JumpStart on HyperPod Inference when the user wants the AWS-managed
container and model configuration with Kubernetes-native infrastructure
control.

Read the selected model sample before giving commands. Obtain model IDs,
versions, supported instance types, request schemas, timeouts, and cleanup
behaviour from that sample.

## Real-time endpoint workflow

1. Confirm the AWS identity, Region, quota, capacity, model variant, instance,
   cost ceiling, expiry, endpoint name, and cleanup owner.
2. Use the sample's lifecycle entry point when it provides one.
3. Refuse to overwrite or delete an existing endpoint unless ownership and the
   requested mutation are explicit.
4. Use independent download, container startup, endpoint deployment, request,
   and cleanup deadlines when the sample supports them.
5. Smoke test with one bounded request.
6. Run cleanup from `finally` for disposable deployments.

If deployment fails before the sample proves resource ownership, do not delete
resources by name. Inspect the endpoint, endpoint configuration, and model.
Report any partial resource that can continue to incur cost.

## JumpStart on HyperPod

Use the current HyperPod sample's configuration, validation, deployment, status,
invocation, and cleanup workflow. A semantic model version is reproducible;
`latest` intentionally resolves the latest available version at deployment
time. State which behaviour the user selected.
