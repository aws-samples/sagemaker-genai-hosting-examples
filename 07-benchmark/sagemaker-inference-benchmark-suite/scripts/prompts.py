"""
Use-case specific prompt datasets for benchmarking.
Designed for prefix caching evaluation — all prompts within a use case share
the same system prompt so that KV cache for the system prompt is reused.

3 use cases:
  1. Multi-turn chat: shared system prompt, increasing conversation depth
  2. Tool calling: shared system prompt with tool definitions, varying queries
  3. Long context: shared system prompt, different long documents as user input

Target token counts (approx):
  - System prompt: ~400-500 tokens (cacheable)
  - User input: ~500-600 tokens
  - Output: ~500-600 tokens (max_tokens=600)
"""

# ==============================================================================
# SHARED SYSTEM PROMPTS (one per use case, cached across all requests)
# ==============================================================================

CHAT_SYSTEM_PROMPT = """You are a senior software engineering assistant with deep expertise in distributed systems, cloud architecture, databases, and modern web development. You help developers solve complex technical problems by providing clear, actionable advice with code examples when appropriate.

Guidelines:
- Provide specific, production-ready solutions rather than generic advice
- Include relevant code snippets in the appropriate language
- Consider edge cases, error handling, and scalability
- When comparing approaches, explain trade-offs clearly
- Reference established patterns and best practices
- If a question is ambiguous, address the most likely interpretation
- Keep responses focused and well-structured with clear sections

You have expertise in: Python, TypeScript, Go, Rust, Java, SQL, AWS, Kubernetes, Docker, React, FastAPI, Django, PostgreSQL, Redis, Kafka, gRPC, GraphQL, CI/CD, observability, and security best practices."""

TOOL_SYSTEM_PROMPT = """You are a helpful AI assistant with access to the following tools. When you need to perform an action, call the appropriate tool. You may chain multiple tool calls to complete complex tasks. Always explain your reasoning before making tool calls.

Available tools:

1. search_web(query: str) -> list[dict]
   Search the web for current information. Returns results with 'title', 'url', 'snippet'.

2. get_weather(city: str, country_code: str) -> dict
   Get current weather conditions. Returns 'temperature_c', 'humidity_pct', 'condition', 'wind_kmh'.

3. calculate(expression: str) -> float
   Evaluate a mathematical expression safely. Supports standard math operations.

4. get_stock_price(symbol: str) -> dict
   Get real-time stock data. Returns 'price_usd', 'change_pct', 'market_cap_b', 'volume'.

5. send_email(to: str, subject: str, body: str) -> bool
   Send an email notification. Returns True on success.

6. create_calendar_event(title: str, date: str, time: str, duration_min: int, attendees: list[str]) -> dict
   Create a calendar event. Returns event details with confirmation ID.

7. query_database(sql: str) -> list[dict]
   Execute a read-only SQL query against the company database. Available tables:
   - orders(id, customer_id, total_amount, status, created_at, region)
   - customers(id, name, email, segment, lifetime_value, signup_date)
   - products(id, name, category, price, cost, stock_count)
   - order_items(id, order_id, product_id, quantity, unit_price)

8. translate_text(text: str, source_lang: str, target_lang: str) -> str
   Translate text between languages. Supports en, es, fr, de, ja, ko, zh.

9. run_shell_command(command: str, server: str) -> dict
   Execute a shell command on a remote server. Returns 'stdout', 'stderr', 'exit_code'.

10. create_jira_ticket(project: str, title: str, description: str, priority: str, assignee: str) -> dict
    Create a Jira ticket. Returns ticket key and URL.

When calling tools, use this JSON format:
{"tool": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}"""

LONG_CONTEXT_SYSTEM_PROMPT = """You are an expert analyst who reads documents carefully and provides thorough, well-structured answers. When answering questions about a document:
- Quote relevant sections to support your analysis
- Organize your response with clear headings and bullet points
- Distinguish between what the document explicitly states vs your inferences
- If the document lacks information to fully answer, say so clearly
- Provide actionable recommendations when appropriate"""

# ==============================================================================
# USE CASE 1: Multi-turn Chat
# All share CHAT_SYSTEM_PROMPT. Conversations of increasing depth.
# Pattern: system → user → assistant → user → assistant → ... → user (final turn)
# The prefix (system + prior turns) gets progressively longer and more cacheable.
# ==============================================================================

MULTITURN_CHAT_PROMPTS = [
    # Conversation depth 1 (short prefix, ~400 system + ~200 user = ~600 input)
    {
        "messages": [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": "I'm building a REST API with FastAPI and need to implement rate limiting. We expect around 10,000 requests per minute at peak. What's the best approach? Should I use middleware, a decorator pattern, or an external service like Redis? We're running on Kubernetes with 8 replicas behind an ALB."},
        ],
        "max_tokens": 600,
        "temperature": 0.7,
    },
    # Conversation depth 2 (system + 2 prior messages cached, ~400+400=~800 cached)
    {
        "messages": [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": "I'm designing a notification system that needs to handle email, SMS, push notifications, and in-app messages. We have about 2 million active users and need to send ~50 million notifications per day. What architecture would you recommend?"},
            {"role": "assistant", "content": "For a notification system at that scale, I recommend an event-driven architecture with these components:\n\n**Core Architecture:**\n- **Event Bus (Kafka/SQS)**: All notification triggers go through a central event bus for decoupling\n- **Notification Service**: Consumes events, applies user preferences, templates, and routes to channel-specific workers\n- **Channel Workers**: Separate worker pools for email (SES), SMS (SNS/Twilio), push (FCM/APNs), and in-app\n- **Priority Queue**: Separate queues for transactional (password reset) vs marketing notifications\n\n**Key Design Decisions:**\n1. Use a notification preferences table to let users control channels per notification type\n2. Implement batching for non-urgent notifications (digest emails)\n3. Rate limit per-user to prevent notification fatigue\n4. Use dead letter queues for failed deliveries with exponential backoff retry"},
            {"role": "user", "content": "Good foundation. Now I'm worried about deliverability and monitoring. How do I track whether notifications actually reached users? For email specifically, we're seeing 15% of our emails going to spam. And for push notifications, our delivery rate is only 60%. How should I set up monitoring and improve these numbers?"},
        ],
        "max_tokens": 600,
        "temperature": 0.7,
    },
    # Conversation depth 3 (system + 4 prior messages cached, ~400+800=~1200 cached)
    {
        "messages": [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": "We're migrating our monolithic Django application to microservices. The monolith handles user management, orders, inventory, payments, and reporting. Where should we start the decomposition?"},
            {"role": "assistant", "content": "Start with the **Strangler Fig Pattern** — don't rewrite everything at once. Here's my recommended order:\n\n**Phase 1: Extract Payments** (highest business risk, clearest boundary)\n- Payments has the clearest domain boundary and the most to gain from independent scaling\n- Create a Payment Service with its own database\n- Use an anti-corruption layer in the monolith that proxies to the new service\n\n**Phase 2: Extract Inventory** (most performance-sensitive)\n- Inventory needs real-time accuracy and can benefit from event sourcing\n- Use Kafka for inventory change events that the monolith consumes\n\n**Phase 3: Extract Orders** (depends on Payments + Inventory)\n- Orders orchestrates between services using the Saga pattern\n- Start with choreography-based sagas, move to orchestration if complexity grows\n\n**Phase 4: User Management** (least urgent, most coupled)\n- Extract last because everything depends on it\n- Use an API Gateway with JWT tokens to decouple authentication"},
            {"role": "user", "content": "Makes sense. Let's focus on Phase 1. We currently process payments through Stripe, and our Django models have Payment, Refund, and Subscription tables with about 40 million rows. How do we migrate the data without downtime?"},
            {"role": "assistant", "content": "Zero-downtime data migration requires a careful multi-step approach:\n\n**Step 1: Dual Write (Week 1-2)**\n- Deploy the new Payment Service alongside the monolith\n- Modify the monolith to write to BOTH the old Django tables and the new service\n- New service writes to its own PostgreSQL database\n- Compare writes for consistency (shadow mode)\n\n**Step 2: Backfill Historical Data (Week 2-3)**\n- Run a batch migration job for the 40M historical rows\n- Use cursor-based pagination, process in chunks of 10K\n- Track progress with a migration checkpoint table\n- Verify row counts and checksums match\n\n**Step 3: Read Migration (Week 3-4)**\n- Gradually shift reads from monolith to new service using feature flags\n- Start with 1% traffic, monitor latency and error rates\n- Ramp up to 100% over a week\n\n**Step 4: Cut Write Path (Week 4-5)**\n- Stop dual writes, all writes go through new service only\n- Keep old tables as read-only backup for 30 days\n- Drop old tables after validation period"},
            {"role": "user", "content": "During the dual-write phase, how do we handle the case where a write succeeds in the monolith but fails in the new service? We can't afford to lose payment records. Also, what happens to in-flight Stripe webhooks during the migration? We process about 500 webhooks per minute."},
        ],
        "max_tokens": 600,
        "temperature": 0.7,
    },
]

# ==============================================================================
# USE CASE 2: Tool Calling
# All share TOOL_SYSTEM_PROMPT with identical tool definitions.
# System prompt (~500 tokens) is cached across all requests.
# ==============================================================================

TOOL_CALLING_PROMPTS = [
    {
        "messages": [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": "I need to prepare for a business trip to Tokyo next week. Can you check the weather forecast, find the current USD to JPY exchange rate, and draft an email to my team (team@company.com) about the trip schedule? My meeting is on Wednesday at 2pm local time with Tanaka-san from Sony. Also create a calendar event for it."},
        ],
        "max_tokens": 600,
        "temperature": 0.7,
    },
    {
        "messages": [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": "Our Q1 revenue seems lower than expected. I need a thorough analysis: 1) Query total revenue by region for Q1 2026 vs Q1 2025, 2) Find the top 5 underperforming product categories by revenue decline, 3) Calculate the year-over-year growth rate for each region, and 4) If any region dropped more than 10%, create a high-priority Jira ticket for the revenue team to investigate."},
        ],
        "max_tokens": 600,
        "temperature": 0.7,
    },
    {
        "messages": [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": "We're getting alerts that our payment-service in production is returning 500 errors at 15% rate for the last 20 minutes. The service was deployed 45 minutes ago with version 2.4.1. Please investigate: check the error logs, look at CPU and memory metrics for the last hour, check if the database connection pool is saturated, and if the issue is from the new deployment, roll back to version 2.3.8. Send a notification to #incidents channel with your findings."},
        ],
        "max_tokens": 600,
        "temperature": 0.7,
    },
]

# ==============================================================================
# USE CASE 3: Long Context Document QA
# All share LONG_CONTEXT_SYSTEM_PROMPT (~100 tokens cached).
# Each request includes a different long document (~2000-3000 tokens) in the user message.
# Tests: long prefill + cached system prompt + varied context.
# ==============================================================================

LONG_CONTEXT_DOC_1 = """The following is a technical architecture document for review.

# Distributed Payment Processing Platform — Architecture Design Document

## 1. System Overview

The Distributed Payment Processing Platform (DPPP) is designed to handle financial transactions at scale, supporting 50,000 transactions per second at peak load with 99.999% availability. The system processes credit card payments, bank transfers, digital wallet transactions, and cryptocurrency payments across 45 countries.

## 2. Core Architecture

### 2.1 Transaction Flow
1. **Ingestion Layer**: API Gateway receives payment requests, validates schema, assigns transaction ID
2. **Authentication**: mTLS + OAuth2 token validation, rate limiting per merchant
3. **Risk Engine**: Real-time fraud detection using ML models (p99 < 50ms)
4. **Routing**: Intelligent payment processor selection based on cost, success rate, and latency
5. **Processing**: Async transaction execution with exactly-once semantics
6. **Settlement**: Daily batch settlement with reconciliation checks

### 2.2 Data Architecture
- **Hot Storage**: Redis Cluster (16 nodes) for active transactions, session data, rate limiting
- **Warm Storage**: PostgreSQL with Citus extension (sharded by merchant_id, 128 shards)
- **Cold Storage**: S3 + Athena for historical analytics, 7-year retention per PCI-DSS
- **Event Store**: Kafka (48 partitions) for transaction events, 30-day retention
- **Search**: Elasticsearch for transaction lookup, merchant dashboards

### 2.3 Consistency Model
We use the Saga pattern for distributed transactions:
- **Payment Saga**: authorize → capture → settle → reconcile
- **Refund Saga**: validate → reverse_auth → credit → notify
- Each step is idempotent with compensation actions for rollback
- Saga orchestrator persists state in PostgreSQL with row-level locking

## 3. Infrastructure

### 3.1 Compute
- **Transaction Workers**: 200 containers on EKS (c7g.4xlarge), auto-scaling 100-400
- **Risk Engine**: 32 GPU instances (g5.2xlarge) for ML inference
- **API Gateway**: Kong on 16 t3.2xlarge instances behind NLB

### 3.2 Network
- Multi-region active-active in us-east-1 and eu-west-1
- AWS Global Accelerator for latency-based routing
- VPC peering between regions, Transit Gateway for partner networks
- Private endpoints for all AWS services (S3, SQS, etc.)

### 3.3 Security
- PCI-DSS Level 1 compliant
- All data encrypted at rest (AES-256) and in transit (TLS 1.3)
- HSM-backed key management for card data encryption
- Tokenization: card numbers never stored, only tokens
- SOC 2 Type II audit annually

## 4. Reliability

### 4.1 Failure Modes
| Failure | Impact | Mitigation | RTO |
|---------|--------|-----------|-----|
| Single AZ failure | None | Multi-AZ deployment | 0s |
| Region failure | 50% capacity | Active-active failover | < 30s |
| Database failure | Degraded | Multi-AZ RDS + read replicas | < 60s |
| Kafka failure | Queued txns | Multi-AZ MSK + dead letter | < 120s |
| Redis failure | Cache miss | Cluster mode + sentinel | < 10s |

### 4.2 Observability
- **Metrics**: Prometheus + Grafana (15s scrape interval)
- **Traces**: Jaeger distributed tracing (100% sampling for errors, 1% for success)
- **Logs**: Structured JSON → Fluentd → Elasticsearch (30-day hot, 1-year cold)
- **Alerts**: PagerDuty integration with escalation policies

## 5. Performance Requirements

| Metric | Target | Current |
|--------|--------|---------|
| p50 latency | < 100ms | 45ms |
| p99 latency | < 500ms | 280ms |
| p99.9 latency | < 2s | 1.1s |
| Throughput | 50K TPS | 62K TPS |
| Availability | 99.999% | 99.9995% |
| Error rate | < 0.01% | 0.003% |

## 6. Cost Analysis (Monthly)
- Compute: $180,000 (EKS workers + GPU instances)
- Database: $45,000 (RDS + ElastiCache + Elasticsearch)
- Network: $32,000 (data transfer + Global Accelerator)
- Storage: $8,000 (S3 + EBS)
- Security: $15,000 (HSM + WAF + Shield Advanced)
- **Total**: $280,000/month for 50K TPS capacity"""

LONG_CONTEXT_DOC_2 = """The following is a machine learning operations report for review.

# Q1 2026 MLOps Platform Performance Report

## Executive Summary

This report covers the performance, reliability, and cost metrics of our ML platform for Q1 2026. The platform serves 47 production models across recommendation, search ranking, fraud detection, and content moderation workloads. Total inference volume reached 12.8 billion requests in Q1, up 34% from Q4 2025.

## 1. Model Serving Infrastructure

### 1.1 Serving Stack
- **Framework**: TorchServe 0.9.0 on Amazon SageMaker endpoints
- **Hardware**: Mix of ml.g5.2xlarge (A10G), ml.inf2.xlarge (Inferentia2), ml.g7e.2xlarge (Blackwell)
- **Auto-scaling**: Target-tracking on InvocationsPerInstance metric, min/max per model class
- **Load Balancing**: SageMaker built-in with sticky sessions for A/B tests

### 1.2 Model Inventory

| Model Class | Count | Avg Size | Hardware | Monthly Cost |
|-------------|-------|----------|----------|-------------|
| Recommendation | 12 | 2.1B params | g5.2xlarge | $89,000 |
| Search Ranking | 8 | 850M params | inf2.xlarge | $34,000 |
| Fraud Detection | 6 | 450M params | g5.xlarge | $28,000 |
| Content Moderation | 5 | 3.2B params | g5.4xlarge | $52,000 |
| Language Models | 4 | 32B params | g7e.2xlarge | $41,000 |
| Vision Models | 7 | 1.5B params | g5.2xlarge | $38,000 |
| Embeddings | 5 | 400M params | inf2.xlarge | $12,000 |

### 1.3 Performance Metrics (Q1 2026)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| p50 latency (all models) | < 50ms | 23ms | ✅ |
| p99 latency (all models) | < 200ms | 145ms | ✅ |
| Availability | 99.95% | 99.97% | ✅ |
| Throughput capacity | 200K RPS | 248K RPS | ✅ |
| GPU utilization | > 70% | 64% | ⚠️ Below target |
| Cost per 1M inferences | < $0.50 | $0.42 | ✅ |

## 2. Training Infrastructure

### 2.1 Training Compute
- **On-demand**: 8x p5.48xlarge (8x H100 per node) for large model training
- **Spot**: Up to 32x p4de.24xlarge for hyperparameter sweeps
- **Storage**: FSx for Lustre (120TB) for training data, 2.4 GB/s throughput
- **Orchestration**: SageMaker Training with managed spot training (60% cost reduction)

### 2.2 Training Runs (Q1 2026)

| Model | Runs | Avg Duration | Avg Cost | Best Metric Improvement |
|-------|------|-------------|----------|----------------------|
| RecSys v14 | 23 | 18 hours | $4,200 | +2.3% CTR |
| FraudNet v7 | 45 | 6 hours | $890 | +4.1% precision |
| SearchRank v9 | 31 | 12 hours | $2,100 | +1.8% NDCG |
| ContentMod v5 | 12 | 24 hours | $8,400 | +3.2% accuracy |
| LLM FineTune | 8 | 72 hours | $28,000 | -15% hallucination |
| EmbedModel v3 | 18 | 4 hours | $520 | +5.2% recall@10 |

### 2.3 Training Cost Summary
- Total training spend: $892,000 (Q1 2026)
- Spot savings: $534,000 (37.5% of on-demand equivalent)
- Training efficiency: 78% GPU utilization during training runs
- Wasted compute (failed/abandoned runs): $67,000 (7.5%)

## 3. Model Quality & Monitoring

### 3.1 Data Drift Detection
We monitor feature distributions using Population Stability Index (PSI):
- **Recommendation models**: PSI threshold 0.1, triggered 3 times in Q1 (all resolved by retraining)
- **Fraud models**: PSI threshold 0.05 (stricter), triggered 7 times (2 required manual investigation)
- **Search models**: Stable, no drift detected

### 3.2 A/B Testing Results

| Experiment | Duration | Sample Size | Result | Deployed? |
|-----------|----------|-------------|--------|-----------|
| RecSys v14 vs v13 | 14 days | 2.4M users | +2.3% CTR (p<0.001) | ✅ Yes |
| FraudNet v7 vs v6 | 7 days | 8.1M txns | +4.1% precision, -0.2% recall | ✅ Yes |
| SearchRank v9 vs v8 | 21 days | 5.6M queries | +1.8% NDCG (p<0.01) | ✅ Yes |
| NewEmbed vs OldEmbed | 14 days | 3.2M users | +5.2% recall, -1.1% latency | ✅ Yes |
| LLM v2 vs v1 | 30 days | 800K convos | -15% hallucination, +8% satisfaction | ✅ Yes |
| ContentMod v5 vs v4 | 7 days | 12M items | +3.2% accuracy, same latency | ✅ Yes |

## 4. Incidents

### 4.1 Incident Summary
- **SEV1**: 0 incidents (target: 0) ✅
- **SEV2**: 2 incidents (target: < 3) ✅
  - Jan 15: RecSys endpoint OOM due to memory leak in TorchServe handler (43 min downtime)
  - Mar 2: Fraud model stale predictions due to feature store lag (28 min degraded)
- **SEV3**: 8 incidents (target: < 10) ✅

## 5. Cost Optimization Recommendations

1. **Migrate recommendation models to Inferentia2**: Projected 45% cost reduction ($89K → $49K/month)
2. **Implement model compilation (torch.compile)**: 20-30% latency improvement, enabling smaller instances
3. **Right-size fraud detection endpoints**: Current 64% GPU util suggests over-provisioning
4. **Use SageMaker Savings Plans**: Commit to 1-year plan for stable workloads (30% discount)
5. **Estimated annual savings**: $680,000"""

LONG_CONTEXT_DOC_3 = """The following is a security audit report for review.

# Cloud Infrastructure Security Audit Report — Annual Assessment 2026

## 1. Scope and Methodology

This audit covers the complete AWS infrastructure for Acme Corp's production environment across us-east-1 and eu-west-1 regions. Assessment period: January 1 - March 15, 2026. Methodology follows NIST Cybersecurity Framework, CIS AWS Benchmarks v3.0, and OWASP Top 10 2025.

### 1.1 Assets Audited
- 847 EC2 instances across 12 VPCs
- 234 RDS instances (PostgreSQL, MySQL, Aurora)
- 89 S3 buckets (42 public-facing via CloudFront)
- 156 Lambda functions
- 23 EKS clusters (1,847 pods)
- 45 SageMaker endpoints
- 12 API Gateway REST APIs
- Network infrastructure (Transit Gateway, VPN, Direct Connect)

## 2. Critical Findings

### 2.1 CRITICAL — IAM Over-Privileged Service Accounts (CVSS 9.1)
**Finding**: 23 service accounts have AdministratorAccess or PowerUserAccess policies. 8 of these are used by CI/CD pipelines and have access keys older than 365 days.

**Evidence**:
- `svc-deploy-prod` has AdministratorAccess, last key rotation: 2024-11-03 (497 days ago)
- `svc-data-pipeline` has PowerUserAccess + direct S3 full access
- `svc-monitoring` has ReadOnlyAccess to ALL accounts via cross-account roles

**Risk**: Compromised CI/CD credentials could lead to full account takeover, data exfiltration, or infrastructure destruction.

**Remediation**:
1. Implement least-privilege policies for all service accounts within 30 days
2. Enable automatic key rotation (90-day maximum)
3. Use IAM Roles Anywhere or OIDC federation for CI/CD instead of static keys
4. Add SCP guardrails to prevent service accounts from modifying IAM policies

### 2.2 CRITICAL — Unencrypted Database Connections (CVSS 8.7)
**Finding**: 18 of 234 RDS instances accept unencrypted connections. 6 of these contain PII (names, emails, addresses) and 2 contain financial data (transaction records).

**Evidence**:
- Parameter group `default.postgres14` has `rds.force_ssl = 0` for 12 instances
- MySQL instances in `vpc-legacy-app` use `require_secure_transport = OFF`
- Network capture shows plaintext SQL queries traversing VPC peering connections

**Risk**: Man-in-the-middle attacks could intercept PII and financial data, violating GDPR, SOC2, and PCI-DSS requirements.

**Remediation**:
1. Enable `rds.force_ssl = 1` for all PostgreSQL instances immediately
2. Set `require_secure_transport = ON` for all MySQL instances
3. Update application connection strings to use SSL certificates
4. Monitor for unencrypted connections via VPC Flow Logs analysis

### 2.3 HIGH — S3 Bucket Policy Misconfigurations (CVSS 7.5)
**Finding**: 7 S3 buckets have overly permissive bucket policies. 3 buckets allow `s3:GetObject` from `"Principal": "*"` without CloudFront OAI restriction, exposing internal data.

**Affected Buckets**:
- `acme-internal-docs` — 45,000 internal documents, some containing employee PII
- `acme-ml-training-data` — training datasets with user behavior data
- `acme-backup-2024` — database backups (encrypted at rest, but publicly listable)

### 2.4 HIGH — Kubernetes RBAC Excessive Permissions (CVSS 7.3)
**Finding**: 12 of 23 EKS clusters grant `cluster-admin` role to developer service accounts. Default namespace is used for production workloads in 4 clusters, with no network policies applied.

### 2.5 HIGH — Missing WAF Rules on Public APIs (CVSS 7.0)
**Finding**: 4 of 12 API Gateway endpoints lack AWS WAF integration. These APIs process authentication tokens and user data without rate limiting at the WAF level.

## 3. Medium Findings (Summary)

| # | Finding | CVSS | Affected Assets |
|---|---------|------|----------------|
| 3.1 | CloudTrail not enabled in 2 regions | 6.5 | ap-southeast-1, sa-east-1 |
| 3.2 | Security groups with 0.0.0.0/0 on non-HTTP ports | 6.3 | 34 EC2 instances |
| 3.3 | EBS volumes unencrypted | 6.0 | 89 volumes (2.3TB) |
| 3.4 | Lambda functions with outdated runtimes | 5.8 | 23 functions (Python 3.8, Node 14) |
| 3.5 | Missing MFA on 12 IAM users | 5.5 | 12 console users |
| 3.6 | VPC Flow Logs not enabled | 5.0 | 3 VPCs |
| 3.7 | Secrets in environment variables | 5.0 | 8 ECS task definitions |
| 3.8 | Certificate expiry within 30 days | 4.5 | 6 ACM certificates |

## 4. Compliance Status

| Framework | Status | Issues | Previous Audit |
|-----------|--------|--------|---------------|
| SOC 2 Type II | ⚠️ At Risk | 3 controls failing | Compliant |
| PCI-DSS v4.0 | ❌ Non-compliant | Unencrypted DB connections | Compliant |
| GDPR | ⚠️ At Risk | S3 data exposure, IAM keys | Compliant |
| HIPAA | ✅ Compliant | No issues | Compliant |
| CIS Benchmark | 67% pass | 89 of 268 checks failing | 72% pass |

## 5. Remediation Timeline

| Priority | Finding | Deadline | Owner |
|----------|---------|----------|-------|
| P0 | Force SSL on all databases | 7 days | DBA Team |
| P0 | Rotate all IAM access keys > 90 days | 14 days | Security |
| P1 | Fix S3 bucket policies | 14 days | Platform |
| P1 | RBAC audit for EKS clusters | 21 days | DevOps |
| P1 | Deploy WAF on all public APIs | 21 days | Platform |
| P2 | Enable CloudTrail in all regions | 30 days | Security |
| P2 | Encrypt all EBS volumes | 30 days | DevOps |
| P3 | Update Lambda runtimes | 45 days | Dev Teams |
| P3 | Enable MFA for all users | 14 days | IT |

## 6. Year-over-Year Comparison

| Metric | 2025 Audit | 2026 Audit | Trend |
|--------|-----------|-----------|-------|
| Critical findings | 1 | 2 | ⬆️ Worse |
| High findings | 2 | 3 | ⬆️ Worse |
| Medium findings | 6 | 8 | ⬆️ Worse |
| CIS benchmark pass rate | 72% | 67% | ⬇️ Worse |
| Mean time to remediate (P0) | 5 days | N/A (in progress) | - |
| Assets with known vulnerabilities | 45 | 78 | ⬆️ Worse |

The overall security posture has degraded compared to 2025, primarily due to rapid infrastructure growth (+62% more assets) outpacing security team capacity. Immediate attention is required on the critical and high findings to maintain compliance certifications."""

LONG_CONTEXT_PROMPTS = [
    {
        "messages": [
            {"role": "system", "content": LONG_CONTEXT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Please analyze the following architecture document and answer: What are the top 3 single points of failure in this system, and how would you improve the reliability design? Also evaluate whether the monthly cost of $280K is justified for the stated performance targets.\n\n{LONG_CONTEXT_DOC_1}"},
        ],
        "max_tokens": 600,
        "temperature": 0.7,
    },
    {
        "messages": [
            {"role": "system", "content": LONG_CONTEXT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Review this MLOps report and provide: 1) Which model class has the worst cost-efficiency and why? 2) Prioritize the 5 cost optimization recommendations with estimated effort vs impact. 3) Are there any red flags in the incident data or A/B test results that the team should investigate?\n\n{LONG_CONTEXT_DOC_2}"},
        ],
        "max_tokens": 600,
        "temperature": 0.7,
    },
    {
        "messages": [
            {"role": "system", "content": LONG_CONTEXT_SYSTEM_PROMPT},
            {"role": "user", "content": f"As the CISO reviewing this audit, I need: 1) An executive summary of the most urgent items for the board. 2) A realistic assessment of whether we can maintain SOC2 and PCI-DSS compliance given the current findings. 3) What's missing from the remediation plan? Identify gaps and additional actions needed. 4) Why has our security posture degraded year-over-year and what systemic changes do we need?\n\n{LONG_CONTEXT_DOC_3}"},
        ],
        "max_tokens": 600,
        "temperature": 0.7,
    },
]


# ==============================================================================
# Registry
# ==============================================================================

USE_CASE_PROMPTS = {
    "multiturn_chat": MULTITURN_CHAT_PROMPTS,
    "tool_calling": TOOL_CALLING_PROMPTS,
    "long_context": LONG_CONTEXT_PROMPTS,
}

USE_CASE_DESCRIPTIONS = {
    "multiturn_chat": "Multi-turn Chat (shared system prompt, increasing depth)",
    "tool_calling": "Tool Calling (shared system+tools prompt, varying queries)",
    "long_context": "Long Context Document QA (shared system prompt, different docs)",
}
