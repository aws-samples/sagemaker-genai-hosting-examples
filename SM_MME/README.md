# Legal AI Multi-Adapter Fine-Tuning

A comprehensive SageMaker JumpStart pipeline for fine-tuning Mistral-7B with multiple QLoRA adapters for legal AI tasks, including contract review, legal summarization, and legal Q&A.


### Current
- **Fine-tuning**: 3 domains 
- **Highly Available** architecture
- **Latency**: <5 seconds
- **Volume**: 50K requests/day
- **Accuracy**: >85%

![Architecture]architecture_image.png

## 📋 Overview

This project demonstrates end-to-end fine-tuning and deployment of specialized legal AI models using:
- **Base Model**: Mistral-7B 
- **Technique**: QLoRA (Quantized Low-Rank Adaptation)
- **Infrastructure**: AWS SageMaker with JumpStart
- **Deployment**: Multi-adapter inference components with LMI (Large Model Inference)

## 🎯 Use Cases

### 1. Contract Review
- **Dataset Size**: 18,621 examples
- **Max Length**: 512 tokens (covers 98.4% of examples)
- **Task**: Identify risks, obligations, and key clauses in legal contracts

### 2. Legal Summary
- **Dataset Size**: 17,880 examples
- **Max Length**: 768 tokens (covers 99.3% of examples)
- **Task**: Generate concise summaries of legal documents

### 3. Legal Q&A
- **Dataset Size**: Custom dataset
- **Task**: Answer questions about legal concepts, regulations, and precedents

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│           Base Model: Mistral-7B (S3)               │
│              (4-bit Quantized)                      │
└─────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  QLoRA       │ │  QLoRA       │ │  QLoRA       │
│  Adapter 1   │ │  Adapter 2   │ │  Adapter 3   │
│              │ │              │ │              │
│  Legal Q&A   │ │  Contract    │ │  Legal       │
│              │ │  Review      │ │  Summary     │
└──────────────┘ └──────────────┘ └──────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        ▼
        ┌─────────────────────────────────┐
        │  LMI Multi-Adapter Endpoint     │
        │  (ml.g5.12xlarge - 4xA10G)      │
        └─────────────────────────────────┘
```

## Adapter Descriptions

### contract-review1

Purpose: Analyze, review, and identify risks in contract clauses and legal agreements.

Typical queries:
- "Review this clause: The vendor shall indemnify..."
- "Identify risks in this termination provision"
- "Analyze the non-compete clause for enforceability"

Keyword triggers: review, analyze, clause, indemnification, termination, non-compete,
liability, warranty, force majeure, enforceability, redline

Context signals: hereby agrees, shall not, notwithstanding, in witness whereof, whereas

### legal-qa1

Purpose: Answer general legal questions, explain concepts, and provide legal knowledge.

Typical queries:
- "What is the statute of limitations for breach of contract?"
- "Explain the difference between negligence and gross negligence"
- "What are the requirements for a valid NDA?"

Keyword triggers: what is, what are, explain, define, how does, difference between,
is it legal, rights of, obligations of, penalty for, under which law

This is the default adapter when routing cannot determine intent.

### legal-summary1

Purpose: Summarize legal judgments, case proceedings, and legal documents.

Typical queries:
- "Summarize this court ruling"
- "Provide a brief of this judgment"
- "Key takeaways from this legal proceeding"

Keyword triggers: summarize, summary, key holdings, brief this, main points,
key takeaways, judgment summary, case brief

Context signals: the court held, tribunal, judgment, verdict, ruling,
petitioner, respondent, appellant


## 🚀 Getting Started

### Prerequisites

```python
# Required versions
sagemaker >= 2.219.0
transformers == 4.36.0
pytorch == 2.1.0
python >= 3.10
```

### AWS Setup

```python
import sagemaker
import boto3

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
region = sagemaker_session.boto_region_name
bucket = sagemaker_session.default_bucket()
```

## 📊 Training Configuration


### DATA FORMAT REQUIREMENTS

Data MUST be JSON/JSONL with these fields:

{
  "instruction": "What is the penalty for breach of contract?",
  "context": "Section 73 of the Indian Contract Act...",
  "output": "The penalty for breach of contract..."
}


### Hyperparameters

| Task | Batch Size | Max Length | LoRA Rank | Epochs | Training Time |
|------|------------|------------|-----------|--------|---------------|
| Contract Review | 16 | 512 | 32 | 3 | ~35 min |
| Legal Summary | 10 | 768 | 32 | 3 | ~45 min |
| Legal Q&A | 16 | 512 | 32 | 3 | ~35 min |

### TRAINING METHOD (QLORA)
1. The python script leverages QLORA (4-bit quantized base + LORA) to further cost optimize the training process
2. The base model is never modified. Its frozen and compressed to 4-bit
3.  Double quantization provides additional memory savings with negligible quality loss (saves ~0.4 bits/param extra)
4. "NF4" gives the best possible quality when compressing the model, less information is lost and better model quality after quantization
5. R=32, more expressive adapters, more trainable params, better quality but more memory
6. lora_alpha=64, this controls how much adapters influence the output
7. target_modules=7 which is comprehensive to cover all attention layers
8. lora_dropout=0.05, light regularization to prevent overfitting


### ADDITIONAL POINTS

1. The token length distribution is set to 512, based on the synthetic data, it is important to check the token length distribution of your data to avoid truncation
2. W&B logging is being used in this case. Other 3rd party tools like ML flow can be configured as well

### Training Infrastructure

- **Instance Type**: `ml.g5.48xlarge` (8x A10G GPUs, 192GB GPU RAM)
- **Distribution**: PyTorch Distributed (torch_distributed)
- **Framework**: HuggingFace Transformers 4.36.0
- **Monitoring**: Weights & Biases (W&B)

### Parallel Training

All three adapters are trained in parallel to optimize time:

```python
# Launch contract review training
est_contract.fit({"model": model_uri, "training": training_data_s3}, wait=False)

# Launch legal summary training
est_summary.fit({"model": model_uri, "training": training_data_s3}, wait=False)

# Launch legal Q&A training
est_qa.fit({"model": model_uri, "training": training_data_s3}, wait=False)
```

## Evaluation

### LLM-as-a-Judge Evaluation

Uses **AWS Bedrock Claude Haiku** (Haiku is being used for demonstration purposes, you can and should use Sonnet or any latest LLM) to evaluate model outputs on:
- **Correctness**: Accuracy of information (threshold: 4/5)
- **Completeness**: Coverage of key points (threshold: 4/5)
- **No Hallucination**: Factual accuracy (threshold: 4/5)

### Evaluation Pipeline

1. Load base Mistral-7B model (4-bit quantized)
2. Load QLoRA adapter on top
3. Load test dataset
4. Generate predictions for each test example
5. Save predictions as JSON for review
6. Run LLM judge evaluation using Bedrock

### Sample Results

```
Legal Q&A V2 Evaluation:
✅ Pass Rate: 75.6% (34/45 examples)
📊 Avg Scores:
   - Correctness: 4.1/5
   - Completeness: 4.0/5
   - No Hallucination: 4.6/5
```

Please note evaluation is being done on a sample test set. Ideally you should either procure a test set or divide the original dataset into test and train. 

## 🌐 Deployment

### Multi-Adapter Inference Components

The deployment uses SageMaker Inference Components for efficient multi-adapter serving:

A production-ready multi-adapter inference system built on Amazon SageMaker, serving multiple
fine-tuned LoRA adapters on a single endpoint with intelligent query routing. The system uses
Mistral-7B-Instruct-v0.2 as the base model with three specialized legal LoRA adapters, routed
through an AWS Lambda function that analyzes incoming queries and selects the appropriate adapter.

The system implements Method 2 (Base IC + Payload Routing) for SageMaker invocation, enabling
continuous batching across all adapters for optimal throughput. Performance testing showed a
6x throughput improvement over the alternative Method 1 (Direct Adapter IC Routing) approach

## Optional Enhancement: Intelligent Routing (AWS Lambda Function)

All requests are routed through the Base Inference Component. The adapter is specified in the request payload body, not through separate Inference Component routing. This enables the LMI-Dist engine to batch requests across different adapters in the same forward pass, resulting in significantly higher throughput under concurrent load.

## Routing Strategies

### Strategy 1: Keyword-Based Routing

- Method: Pattern matching against predefined keyword lists
- Cost: Free (no external API calls)
- Latency: < 1ms
- Accuracy: Good for well-structured queries with clear intent signals

The router maintains a dictionary of keywords and context signals for each adapter.
Each keyword match scores 2 points. Each context signal match scores 1 point. The adapter
with the highest score wins, provided it meets the confidence threshold.

A "strong" match requires the top score to be at least 2x the second-highest score.
Otherwise the match is classified as "weak" and may trigger fallback to Bedrock in hybrid mode.


### Strategy 2: Bedrock LLM Classifier

- Method: Claude 3 Haiku (You can choose any model here) classification via Amazon Bedrock
- Cost: ~$0.0003 per call (0.25/1M input tokens, 1.25/1M output tokens)
- Latency: 200-500ms
- Accuracy: High, handles ambiguous and complex queries

Sends the query to Claude 3 Haiku (Recommended to use the latest model) with a structured classification prompt.
The model returns one of three category names. Exact matches are accepted directly.
Fuzzy matches (category name found within response text) are accepted with lower confidence.

### Strategy 3: Hybrid (Default, Recommended)

- Method: Keywords first, Bedrock only when ambiguous
- Cost: Free for ~80% of requests, ~$0.0003 for remaining ~20%
- Latency: < 1ms for keyword-resolved, 200-500ms for Bedrock fallback
- Accuracy: Highest overall

The hybrid strategy attempts keyword routing first. If the keyword match is strong
(method = "keyword" and confidence >= 4), the result is used directly without calling Bedrock.
For weak or zero-confidence keyword matches, the request is escalated to Bedrock for classification.

This approach minimizes cost and latency while maintaining high routing accuracy.

---
## Pre-requisistes for setting up the Lambda function for intelligent routing:

The lambda function must have the following permissions:

{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "sagemaker:InvokeEndpoint",
            "Resource": "arn:aws:sagemaker:us-east-2:569202655535:endpoint/legal-lmi-multi-adapter-ep1"
        },
        {
            "Effect": "Allow",
            "Action": "sagemaker:InvokeEndpointWithResponseStream",
            "Resource": "arn:aws:sagemaker:us-east-2:569202655535:endpoint/legal-lmi-multi-adapter-ep1"
        },
        {
            "Effect": "Allow",
            "Action": "bedrock:InvokeModel",
            "Resource": "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"
        }
    ]
}

## Lambda Function Setup
Step 1: Create the IAM Role

    1. Open the AWS Console and navigate to IAM > Roles
    2. Click "Create role"
    3. Select "AWS service" as the trusted entity type
    4. Select "Lambda" as the use case
    5. Click "Next"
    6. Attach the following managed policy (AWSLambdaBasicExecutionRole (for CloudWatch Logs)
    7. Click "Next", name the role legal-router-lambda-role, and click "Create role"
    8. Open the newly created role and click "Add inline policy"
    9. Switch to the JSON editor and paste the following policy and then click create policy:
        {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "SageMakerInvoke",
            "Effect": "Allow",
            "Action": [
                "sagemaker:InvokeEndpoint",
                "sagemaker:InvokeEndpointWithResponseStream"
            ],
            "Resource": "arn:aws:sagemaker:us-east-2:YOUR_ACCOUNT_ID:endpoint/legal-lmi-multi-adapter-ep*"
        },
        {
            "Sid": "BedrockInvoke",
            "Effect": "Allow",
            "Action": "bedrock:InvokeModel",
            "Resource": "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"
        }
    ]
}

Step 2: Create the Lambda Function

    1. Open the AWS Console and navigate to Lambda
    2. Click "Create function"
    3. Select "Author from scratch"
    4. Configure:
        Function name: legal-multi-adapter-router
        Runtime: Python 3.12
        Architecture: x86_64
        Execution role: "Use an existing role" > select legal-router-lambda-role
    5. Click "Create function"

Step 3: Deploy the Code

    1. In the Lambda function page, scroll to the "Code source" section
    2. Open the file lambda_function.py in the inline editor
    3. Copy the entire contents of lambda_router.py from this repository
    4. Paste into the editor
    5. Click "Deploy"

Step 4: Configure Environment Variables

    1. In the Lambda function page, click the "Configuration" tab
    2. Click "Environment variables" in the left sidebar
    3. Click "Edit"
    4. Add the following variables (change the names if you rename them):
    CONTRACT_IC contract-review1
    ENDPOINT_NAME legal-lmi-multi-adapter-ep1
    LEGAL_QA_IC legal-qa1
    SUMMARY_IC legal-summary1
    5. Click "Save"

Step 5: Configure Timeout and Memory

    1. In the "Configuration" tab, click "General configuration"
    2. Click "Edit"
    Set the following:
        Memory: 256 MB
        Timeout: 1 minute 0 seconds
    3. Click "Save"

Step 6: Test the Lambda Function

    1. In the Lambda function page, click the "Test" tab
    2. Create a new test event with the following JSON (feel free to edit based on your use case):

    {
    "question": "Help me understand the implications of this employment agreement",
    "context": "Employee agrees to a 2-year non-compete clause within 100 miles of company headquarters. Upon termination, employee shall return all confidential materials within 5 business days.",
    "routing_strategy": "hybrid",
    "max_tokens": 156
    }

    3. Click "Test"
    4. Verify the response contains a 200 status code
---

## API Reference

### Request Format

```json
{
    "question": "string (required) — The legal query",
    "context": "string (optional) — Supporting document, clause, or context text",
    "routing_strategy": "string (optional) — One of: keyword, bedrock, hybrid. Default: hybrid",
    "max_tokens": "integer (optional) — Maximum tokens to generate. Default: 384",
    "temperature": "float (optional) — Sampling temperature. Default: 0.1"
}


## References

- [SageMaker JumpStart Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LMI Container Documentation](https://docs.djl.ai/docs/serving/serving/docs/lmi/index.html)
- [Mistral AI](https://mistral.ai/)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## 📄 License

This project is for demonstration purposes. AI-generated datasets are used for educational purposes only.

## Important Notes

- **Dataset**: AI-generated for demonstration purposes only
- **Production Use**: Validate with real legal data and consult legal experts
- **Compliance**: Ensure compliance with local regulations for AI in legal applications
- **Monitoring**: Continuously monitor model outputs for accuracy and bias


---

**Last Updated**: March 2026
**Notebook Version**: 1.0
**SageMaker SDK**: 2.219.0
