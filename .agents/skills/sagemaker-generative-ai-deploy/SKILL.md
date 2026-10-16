---
name: sagemaker-generative-ai-deploy
description: Plan, deploy, test, benchmark, and clean up generative AI models on Amazon SageMaker. Use when choosing GPU hardware, checkpoint precision, capacity, SageMaker JumpStart, real-time endpoints, or HyperPod Inference. Uses Reflection Beam as a worked example while keeping release-specific commands in model samples.
---

# SageMaker generative AI deployment

Help the user choose the simplest supported Amazon SageMaker deployment for
their model and workload. Treat each model sample as the source of truth for
release-specific identifiers, runtime versions, manifests, commands, and
cleanup procedures.

Read the references required for the task:

- [Hardware and capacity](references/hardware-and-capacity.md)
- [SageMaker JumpStart](references/jumpstart.md)
- [SageMaker HyperPod Inference](references/hyperpod.md)
- [Benchmarking](references/benchmarking.md)
- [Reflection Beam worked example](references/examples/reflection-beam.md)

## Start with the deployment interview

Collect these inputs before recommending a manifest or command:

1. **Model and sample** — model, checkpoint variants, licence or access
   requirements, and the relevant repository sample.
2. **Workload** — context length, output length, concurrency, latency target,
   availability target, traffic pattern, and whether custom runtime settings
   are required.
3. **Hardware** — existing or preferred accelerator, instance type, GPU count,
   node count, and Region.
4. **Capacity product** — existing capacity, on-demand capacity, or a SageMaker
   training plan.
5. **Serving path** — JumpStart real-time endpoint, JumpStart on HyperPod
   Inference, or a custom runtime on HyperPod Inference.
6. **Operating limits** — deployment window, expected cost, maximum spend,
   non-production expiry, and cleanup owner.
7. **Existing infrastructure** — AWS account or profile, quota, networking,
   IAM permissions, HyperPod cluster, Kubernetes namespace, storage, and model
   access.

Combine questions when the user already supplied the answer. Separate confirmed
facts from assumptions. Never invent capacity availability, model identifiers,
instance mappings, image digests, parser names, runtime versions, GPU counts,
or performance results.

## Make the deployment decision

### Choose checkpoint, hardware, and topology together

Treat the following as one compatibility tuple:

- checkpoint format and immutable revision;
- accelerator, instance type, GPUs per node, and node count;
- runtime and image revision;
- tensor, pipeline, data, or expert parallel layout;
- context length, KV cache, batching, and runtime memory overhead.

Do not choose precision only from the accelerator's supported data types. Check
that the model and runtime fit the complete supported topology. Calculate
steady GPU count as GPUs per replica multiplied by replicas. Include temporary
rollout overlap when calculating peak GPUs and cost.

### Choose the capacity product

- Use **existing capacity** when a compatible endpoint allocation or HyperPod
  instance group already exists.
- Use **on-demand capacity** when quota and supply are available and timing is
  flexible.
- Use a **SageMaker training plan** for a scheduled reservation. A purchase or
  extension is prepaid and final: it cannot be cancelled, and the purchased
  capacity configuration cannot be changed. Show the exact AWS account, target
  resource, Region, Availability Zone, instance type, quantity, window,
  offering ID, currency, and upfront price before purchase or extension.

### Choose the serving path

| Serving path | Use it when |
|---|---|
| JumpStart real-time endpoint | The user wants a managed model package, HTTPS endpoint, and invocation API without operating Kubernetes. |
| JumpStart on HyperPod Inference | The user wants an AWS-managed container and model configuration with Kubernetes-native cluster control. |
| Custom runtime on HyperPod Inference | The user needs a specific inference runtime or configuration, such as custom arguments, parsers, quantization, or parallelism. |

Start with JumpStart unless the workload requires a specific runtime or
Kubernetes-native infrastructure control.

## Use the repository as the source of truth

Locate the selected model sample and read its README before giving commands.
Use the sample's current identifiers, supported versions, request schema,
manifests, validation, timeouts, and cleanup instructions.

Do not copy release-specific values into this Skill. Do not direct users to
private or partner-only assets.

## Require approval for mutations

Get current-chat confirmation before:

- purchasing or extending reserved capacity;
- creating, updating, scaling, or deleting an endpoint;
- changing a HyperPod cluster, node group, Kubernetes resource, or Secret;
- mirroring an image or checkpoint;
- running a benchmark that creates cost or changes load or replica count;
- deleting cached weights, images, load balancers, roles, storage, or results.

Show one approval draft:

```text
SAGEMAKER DEPLOYMENT DRAFT
──────────────────────────
Account:          <account or profile>
Region:           <region>
Model/sample:     <model and repository sample>
Checkpoint:       <format and revision>
Hardware:         <instance and accelerator>
GPU footprint:    <steady and peak GPUs>
Capacity:         <existing|on-demand|training plan>
Serving path:     <JumpStart real-time|JumpStart HyperPod|custom HyperPod>
Runtime:          <managed or named runtime>
Resources:        <endpoint, cluster, namespace>
Window:           <start and duration>
Expiry/review:    <non-production TTL or production review date>
Cleanup owner:    <person or team>
Expected change:  <concise mutation>
Estimated cost:   <hourly and total estimate, or explicit unknown>
Maximum spend:    <approved ceiling>
Training plan:    <offering, target, quantity, AZ, fee, or N/A>

ok to proceed, or tell me what to change
```

For a training-plan purchase or extension, `unknown` cost is not acceptable.
Require the exact upfront price and explicit acknowledgement that the purchase
is final. For every non-production deployment, require an expiry and cleanup
owner. A production deployment requires a named owner and review date.

Never print, log, commit, or pass credentials as command-line arguments. Use
environment variables, credential helpers, or existing secret stores. Keep
generated configuration and request files ignored.

## Workflow

1. Complete the deployment interview.
2. Read the selected sample README and current release configuration.
3. Confirm AWS identity, Region, quota, capacity, resource ownership, expiry,
   and cleanup owner.
4. Calculate weight memory, KV cache, runtime overhead, steady GPUs, peak GPUs,
   and expected cost.
5. Recommend one supported checkpoint, hardware, capacity, serving path, and
   topology.
6. Create ignored local configuration using the sample's documented process.
7. Run local validation and a server dry run when supported.
8. Show the deployment draft and get confirmation.
9. Deploy with the sample's bounded lifecycle command rather than applying
   templates directly.
10. Wait for a terminal state. A timeout is unresolved, not complete: continue
    reconciling the named resources or hand off monitoring and cleanup to a
    named owner with a deadline.
11. Run one bounded health check and inference smoke test.
12. Benchmark only after a separate cost and load confirmation.
13. Clean up through the selected sample. Verify deletion before removing
    dependencies, then inventory anything that remains or can incur cost.

## Report the result

Report:

- selected sample, path, checkpoint, instance, GPU count, Region, and capacity;
- model and image revisions for custom runtimes;
- deployment, readiness, and smoke-test status;
- benchmark methodology and limitations when load was approved;
- cleanup status and every resource that remains.

Do not generalise one environment's measurements into a model-wide or
SageMaker-wide performance claim.
