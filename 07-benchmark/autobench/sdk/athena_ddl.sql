-- Benchmarking Initiative: Athena Table DDL
-- Run each statement separately in Athena console (us-east-2)
-- Last updated: May 26, 2026

-- Step 1: Create database
CREATE DATABASE IF NOT EXISTS benchmarking;

-- Step 2: Drop existing table
DROP TABLE IF EXISTS benchmarking.benchmark_metrics;

-- Step 3: Create table with expanded schema
CREATE EXTERNAL TABLE benchmarking.benchmark_metrics (
  job_id STRING,
  model_key STRING,
  model_name STRING,
  model_id STRING,
  concurrency INT,
  input_tokens INT,
  output_tokens INT,
  streaming BOOLEAN,
  duration INT,
  warmup INT,
  dataset STRING,
  instance_type STRING,
  num_gpus INT,
  source_region STRING,
  s3_output STRING,
  `timestamp` STRING,
  vllm_config STRING,
  request_throughput_rps DOUBLE,
  total_token_throughput_tps DOUBLE,
  output_token_throughput_tps DOUBLE,
  request_count DOUBLE,
  ttft_avg_ms DOUBLE,
  ttft_p50_ms DOUBLE,
  ttft_p90_ms DOUBLE,
  ttft_p99_ms DOUBLE,
  itl_avg_ms DOUBLE,
  itl_p50_ms DOUBLE,
  itl_p90_ms DOUBLE,
  itl_p99_ms DOUBLE,
  e2e_latency_avg_ms DOUBLE,
  e2e_latency_p50_ms DOUBLE,
  e2e_latency_p90_ms DOUBLE,
  e2e_latency_p99_ms DOUBLE,
  prefill_tps_avg DOUBLE,
  prefill_tps_p50 DOUBLE,
  e2e_output_token_tps_avg DOUBLE,
  e2e_output_token_tps_p50 DOUBLE,
  e2e_output_token_tps_p90 DOUBLE,
  ttst_p50_ms DOUBLE,
  ttst_p90_ms DOUBLE,
  output_sequence_length_avg DOUBLE,
  input_sequence_length_avg DOUBLE,
  error_request_count DOUBLE,
  osl_mismatch_count DOUBLE,
  benchmark_duration_sec DOUBLE,
  error_rate DOUBLE,
  vllm_version STRING,
  kv_cache_dtype STRING,
  quantization STRING,
  benchmark_tokenizer STRING,
  serving_config STRING,
  aiperf_version STRING,
  raw_aiperf_json STRING
)
PARTITIONED BY (environment STRING, model STRING, workload STRING)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
WITH SERDEPROPERTIES ('ignore.malformed.json' = 'true')
LOCATION 's3://sagemaker-benchmark-<REGION>-<ACCOUNT_ID>/athena/results/'
TBLPROPERTIES ('has_encrypted_data'='false');

-- Step 4: Discover partitions
MSCK REPAIR TABLE benchmarking.benchmark_metrics;
