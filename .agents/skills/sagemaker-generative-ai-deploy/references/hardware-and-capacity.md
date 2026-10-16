# Hardware and capacity

## Size the complete deployment

Collect:

- checkpoint format and size;
- active and total parameter counts when relevant;
- context and output lengths;
- expected concurrency and batching;
- runtime memory overhead;
- KV cache allocation;
- GPUs per node and number of nodes;
- parallelism and replica count.

Do not treat checkpoint size, active parameters, and per-token compute as the
same quantity. Check the selected sample and runtime documentation for the
supported combination.

Calculate:

- steady GPUs = GPUs per replica × steady replicas;
- peak GPUs = steady GPUs + temporary rollout or replacement capacity;
- maximum GPU-hours = peak GPUs × approved deployment window;
- estimated compute, storage, transfer, and supporting-resource cost.

If pricing or capacity cannot be confirmed, state the gap. Do not convert quota
approval into a claim that capacity is currently available.

## Choose capacity

Before deployment:

1. Confirm that the accelerated instance is supported in the Region.
2. Check the relevant SageMaker service quota.
3. Check actual capacity through the selected capacity product.
4. Include time for image transfer, checkpoint download, shard loading,
   runtime initialisation, smoke testing, and cleanup.
5. Confirm the maximum spend and cleanup deadline.

Use existing capacity when it is compatible and available. Use on-demand
capacity when timing is flexible. Use a SageMaker training plan when the
deployment needs a scheduled reservation.

## Training-plan guardrail

A training-plan purchase or extension is prepaid and final. Before any purchase
or extension, show:

- AWS account;
- target resource type and name;
- Region and Availability Zone;
- instance type and quantity;
- start time, end time, and duration;
- offering ID;
- upfront fee and currency.

Require exact current-chat confirmation of those values. Do not proceed with an
unknown fee or an assumed target resource.
