# Cross-Region Inference Router and Instance Pools for Amazon SageMaker AI

This directory contains a notebook that shows how to keep a SageMaker AI
deployment serving traffic when capacity is lost, using two features together:

- **Cross-Region Inference Router**: one router sends traffic to the same model
  deployed in two Regions. It prefers the local Region, fails over to the other
  Region when the local deployment is unhealthy, and fails back when it
  recovers.

- **Instance pools**: one endpoint lists several instance types in priority
  order. If the preferred type has no capacity, SageMaker AI provisions the next
  one instead of failing.

## Notebook

[`inference_router.ipynb`](inference_router.ipynb) deploys `Qwen/Qwen3-4B` in
two Regions and walks through a failover and failback:

- Creates the model in a home Region and a remote Region on the AWS vLLM Deep
  Learning Container
- Creates an endpoint in each Region with `InstancePools` instead of a single
  `InstanceType`, and shows which instance type each Region provisioned
- Deploys one inference component per Region (`ic-<region>`), one GPU each
- Creates an Inference Router with one routing target (`my-model`) whose
  destinations are the two inference component ARNs
- Invokes the router and shows the serving destination and routing reason
- Deletes the home Region inference component to simulate a Regional failure,
  then shows traffic moving to the remote Region
- Recreates the home Region inference component and shows traffic returning
- Deletes all the resources it created in both Regions


## Inference Router Configuration

A router has one or more routing targets. Each target is a logical model name
with a list of destinations, which are inference component ARNs. Clients call
the router with the target name and do not need to know which Region serves the
request.

```python
sm.create_inference_router(
    InferenceRouterName="my-router",
    RoutingTargets=[
        {
            "Name": "my-model",
            "Destinations": [
                {"ResourceArn": home_ic_arn},
                {"ResourceArn": remote_ic_arn},
            ],
        }
    ],
)
```

Requests go through the `sagemaker-inference-router-runtime` client:

```python
resp = rt.invoke_inference_router(
    InferenceRouterName="my-router",
    RoutingTargetName="my-model",
    ContentType="application/json",
    Body=json.dumps(payload),
)
print(resp["InvokedRoutedResourceArn"], resp["RoutingReason"])
```

`describe_inference_router` reports a `RoutingStatus` of `Routable` or
`Unroutable` for each destination. The routing reasons you see in the notebook
are:

| Situation | Serving Region | `RoutingReason` |
|---|---|---|
| Both Regions healthy | Home | `local-preferred` |
| Home Region unhealthy | Remote | `only-healthy` |
| Home Region recovered | Home | `local-preferred` |

After the home inference component is back `InService`, it can take a few
minutes before the router sends traffic to it again.

## Instance Pools Configuration

The production variant lists instance pools with a priority (1 is the most
preferred). `VariantInstanceProvisionTimeoutInSeconds` limits how long
SageMaker AI waits on one type before it tries the next.

```python
"ProductionVariants": [
    {
        "VariantName": "v1",
        "InitialInstanceCount": 1,
        "VariantInstanceProvisionTimeoutInSeconds": 600,
        "InstancePools": [
            {"InstanceType": "ml.g6.2xlarge", "Priority": 1},
            {"InstanceType": "ml.g5.2xlarge", "Priority": 2},
            {"InstanceType": "ml.g6.xlarge",  "Priority": 3},
            {"InstanceType": "ml.g5.xlarge",  "Priority": 4},
            {"InstanceType": "ml.g7.2xlarge", "Priority": 5},
        ],
    },
]
```

Every instance type in the list must be able to run the model. The type that
gets provisioned depends on available capacity and your service quotas, so it
can differ between Regions and between runs.

## Prerequisites

- Two AWS Regions enabled for your account. The notebook uses `us-east-2`
  (home) and `us-west-2` (remote). Change `REGION_HOME` and `REGION_REMOTE` to
  match yours.
- Quota in both Regions for at least one endpoint instance of one of the types
  in the instance pool list
- A SageMaker execution role that can create, describe, invoke and delete
  models, endpoint configurations, endpoints, inference components and
  inference routers
- A Jupyter environment with AWS credentials and a version of `boto3` that
  includes the Inference Router APIs (the notebook upgrades `boto3`)

When you run the notebook outside SageMaker Studio, set `role` in the notebook
to a SageMaker execution role ARN.

## Run the Example

1. Open `inference_router.ipynb` in SageMaker Studio or another Jupyter
   environment.
2. Set `REGION_HOME` and `REGION_REMOTE`, and run the setup cells.
3. Run Steps 1 to 3 to create the models, endpoints and inference components in
   both Regions.
4. Run Step 4 to create the router and send a request through it.
5. Run Step 5 to simulate a failure in the home Region and see failover.
6. Run Step 6 to recover the home Region and see failback.
7. Run the cleanup cells.

## Cost

The notebook creates GPU endpoints in two Regions that bill while they run.
Run the cleanup cells at the end, and check the SageMaker console in both
Regions for endpoints left behind if a run stops early.

## License

This sample is released under the license in the repository root.
