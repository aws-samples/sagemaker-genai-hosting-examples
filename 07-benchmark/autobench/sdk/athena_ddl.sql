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


-- ═══════════════════════════════════════════════════════════════════════
-- Step 5: Create Views (run after table + partitions exist)
-- ═══════════════════════════════════════════════════════════════════════

-- View: Model Configs (one row per unique model configuration)
CREATE OR REPLACE VIEW benchmarking.v_model_configs AS
SELECT DISTINCT
  job_id,
  model_key,
  model_name,
  environment,
  COALESCE(instance_type, 'Managed (BYOM)') AS instance_type,
  COALESCE(CAST(num_gpus AS VARCHAR), '-') AS num_gpus,
  COALESCE(vllm_version, '-') AS vllm_version,
  COALESCE(kv_cache_dtype, 'auto') AS kv_cache_dtype,
  COALESCE(quantization, 'none') AS quantization,
  COALESCE(benchmark_tokenizer, model_name) AS tokenizer,
  dataset
FROM benchmarking.benchmark_metrics
WHERE request_throughput_rps IS NOT NULL;

-- View: Workload Definitions (configured vs actual token lengths)
CREATE OR REPLACE VIEW benchmarking.v_workload_definitions AS
SELECT DISTINCT
  job_id,
  workload,
  input_tokens AS target_input_tokens,
  output_tokens AS target_output_tokens,
  ROUND(input_sequence_length_avg, 0) AS actual_input_tokens_avg,
  ROUND(output_sequence_length_avg, 0) AS actual_output_tokens_avg,
  dataset,
  CASE workload
    WHEN 'multi_turn_chat' THEN 'Copilots, assistants, customer support'
    WHEN 'rag_document_qa' THEN 'Knowledge bases, legal & medical search'
    WHEN 'agent_tool_calling' THEN 'Coding agents, research agents, automation'
    WHEN 'long_context_scaling' THEN 'Prefill stress test'
    WHEN 'production_traffic_mix' THEN 'Aggregate real traffic simulation'
    WHEN 'shared_system_prompt' THEN 'Prefix caching / prompt reuse'
    ELSE 'Unknown'
  END AS use_case
FROM benchmarking.benchmark_metrics
WHERE request_throughput_rps IS NOT NULL;

-- View: Benchmark Metrics (clean fact table for QuickSight)
CREATE OR REPLACE VIEW benchmarking.v_benchmark_metrics_synthetic_data AS
SELECT 
  job_id,
  model_key,
  model_name, 
  environment, 
  COALESCE(instance_type, 'Managed (BYOM)') AS instance_type,
  workload, 
  concurrency,
  dataset,
  vllm_version,
  kv_cache_dtype,
  quantization,
  input_tokens,
  output_tokens,
  request_throughput_rps,
  total_token_throughput_tps,
  output_token_throughput_tps,
  request_count,
  ttft_avg_ms, ttft_p50_ms, ttft_p90_ms, ttft_p99_ms,
  itl_avg_ms, itl_p50_ms, itl_p90_ms, itl_p99_ms,
  e2e_latency_avg_ms, e2e_latency_p50_ms, e2e_latency_p90_ms, e2e_latency_p99_ms, 
  prefill_tps_avg, prefill_tps_p50,
  e2e_output_token_tps_avg, e2e_output_token_tps_p50, e2e_output_token_tps_p90,
  ttst_p50_ms, ttst_p90_ms,
  output_sequence_length_avg,
  input_sequence_length_avg,
  error_rate,
  benchmark_duration_sec,
  num_gpus,
  source_region,
  "timestamp"
FROM benchmarking.benchmark_metrics
WHERE request_throughput_rps IS NOT NULL
  AND (error_rate IS NULL OR error_rate < 1.0)
  AND dataset = 'synthetic';


-- ═══════════════════════════════════════════════════════════════════════
-- Step 6: Export Views to S3 as CSV (for QuickSight)
-- ═══════════════════════════════════════════════════════════════════════
-- Uses timestamped paths to avoid overwrite conflicts.
-- Replace <YYYYMMDD> with today's date (e.g., 20260529).
-- Old exports auto-expire via S3 lifecycle rule (optional).
--
-- IMPORTANT: UNLOAD fails if destination path already has files.
-- Timestamped paths ensure each export goes to a fresh location.
-- No need to clear S3 first.

-- Export: Model Configs
UNLOAD (SELECT * FROM benchmarking.v_model_configs)
TO 's3://sagemaker-benchmark-<REGION>-<ACCOUNT_ID>/quicksight/model_configs/<YYYYMMDD>/'
WITH (format = 'TEXTFILE', field_delimiter = ',', compression = 'NONE');

-- Export: Workload Definitions
UNLOAD (SELECT * FROM benchmarking.v_workload_definitions)
TO 's3://sagemaker-benchmark-<REGION>-<ACCOUNT_ID>/quicksight/workload_definitions/<YYYYMMDD>/'
WITH (format = 'TEXTFILE', field_delimiter = ',', compression = 'NONE');

-- Export: All Benchmark Metrics (clean, all models)
UNLOAD (SELECT * FROM benchmarking.v_benchmark_metrics_synthetic_data)
TO 's3://sagemaker-benchmark-<REGION>-<ACCOUNT_ID>/quicksight/metrics/<YYYYMMDD>/'
WITH (format = 'TEXTFILE', field_delimiter = ',', compression = 'NONE');


-- ═══════════════════════════════════════════════════════════════════════
-- NOTES
-- ═══════════════════════════════════════════════════════════════════════
-- 
-- S3 export structure:
--   s3://sagemaker-benchmark-<REGION>-<ACCOUNT_ID>/quicksight/
--   ├── model_configs/<YYYYMMDD>/           ← v_model_configs CSV
--   ├── workload_definitions/<YYYYMMDD>/    ← v_workload_definitions CSV
--   └── metrics/<YYYYMMDD>/                 ← v_benchmark_metrics CSV
--
-- QuickSight datasets:
--   - Point each dataset at the latest <YYYYMMDD> prefix
--   - UNLOAD creates files like 0000, 0001 (auto-split by Athena)
--   - No column headers in output — configure QuickSight accordingly
--
-- To refresh after new benchmarks:
--   1. Replace <YYYYMMDD> with today's date
--   2. Run the UNLOAD queries in Athena
--   3. Update QuickSight dataset to point at the new prefix
--
-- Cleanup (optional):
--   Set S3 lifecycle rule to expire quicksight/ objects after 30 days
