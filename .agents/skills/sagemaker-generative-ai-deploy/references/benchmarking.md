# Benchmarking

Benchmark only after a bounded health check and inference request succeed.
Require separate approval for the exact endpoint, hardware, replicas, duration,
concurrency, request count, input size, output-token limit, failure threshold,
and expected cost.

## Preflight

Before sending load, show:

- a redacted endpoint identifier or fingerprint;
- workload and sampling settings;
- maximum concurrency;
- total requests, including warm-up requests;
- maximum input and output volume;
- request and overall deadlines;
- steady and peak GPU count;
- expected GPU-hours and cost;
- stop conditions.

Use conservative limits by default. Larger runs require explicit confirmation
of the exact preflight. Stop before later scenarios when warm-up or measured
failures exceed the approved threshold.

Never disable TLS verification for a remote endpoint. Use insecure transport
only for a loopback port-forward. Keep credentials in the environment or an
existing secret store.

## Comparisons

Do not combine results from different:

- checkpoint formats or revisions;
- instance types, GPU counts, nodes, or replicas;
- serving paths or runtimes;
- image revisions or parser settings;
- context lengths, sampling settings, or cache states.

Record request success rate, p50 and p95 time to first token, p50 and p95
end-to-end latency, output-token throughput, total tokens, wall time, and
errors. Store metrics without prompts, generated text, credentials, endpoint
URLs, or unpublished infrastructure details.

Report methodology and limitations. Do not turn one environment's measurements
into a general performance claim.
