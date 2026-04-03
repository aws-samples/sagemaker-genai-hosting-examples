# ═══════════════════════════════════════════════════════
# lambda_router.py — Intelligent Multi-Adapter Router
# Production Version — Base IC + Payload Routing (Batched)
# 🔄 Updated for new endpoint: legal-lmi-multi-adapter-ep1
# ═══════════════════════════════════════════════════════

import json
import boto3
import os
import time

runtime = boto3.client("sagemaker-runtime")
bedrock = boto3.client("bedrock-runtime", region_name="us-east-2")

# 🔄 UPDATED: New endpoint and base IC
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "legal-lmi-multi-adapter-ep1")
BASE_IC = os.environ.get("BASE_IC", "mistral-7b-base-lmi1")
VALID_ADAPTERS = ["legal-qa1", "contract-review1", "legal-summary1"]


# ─── Strategy 1: Keyword-Based Routing (Fast, Free) ───

# 🔄 UPDATED: Adapter names with "1" suffix
KEYWORD_RULES = {
    "contract-review1": {
        "keywords": [
            "review this clause", "review this contract", "analyze this clause",
            "identify issues", "identify risks", "evaluate this provision",
            "indemnification", "termination clause", "non-compete",
            "liability clause", "warranty clause", "force majeure clause",
            "review the following", "clause analysis", "contract risk",
            "enforceability", "draft review", "redline",
        ],
        "context_signals": [
            "hereby agrees", "shall not", "notwithstanding",
            "in witness whereof", "whereas", "party of the first part",
        ],
    },
    "legal-summary1": {
        "keywords": [
            "summarize", "summary", "summarise", "key holdings",
            "brief this", "provide a summary", "concise summary",
            "main points", "key takeaways", "digest this",
            "judgment summary", "case brief", "legal brief",
        ],
        "context_signals": [
            "the court held", "the bench", "the tribunal",
            "judgment", "verdict", "ruling", "order dated",
            "hon'ble", "petitioner", "respondent", "appellant",
        ],
    },
    "legal-qa1": {
        "keywords": [
            "what is", "what are", "explain", "define",
            "how does", "when can", "who can", "difference between",
            "what happens if", "is it legal", "can someone",
            "rights of", "obligations of", "penalty for",
            "section", "article", "under which law",
        ],
        "context_signals": [],
    },
}


def route_by_keywords(question, context=""):
    """Fast keyword-based routing — zero cost, <1ms"""
    text = (question + " " + context).lower()
    
    scores = {}
    for adapter, rules in KEYWORD_RULES.items():
        score = 0
        for kw in rules["keywords"]:
            if kw in text:
                score += 2
        for signal in rules["context_signals"]:
            if signal in text:
                score += 1
        scores[adapter] = score
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_scores[0][1] > 0 and sorted_scores[0][1] >= 2 * max(sorted_scores[1][1], 1):
        return sorted_scores[0][0], "keyword", sorted_scores[0][1]
    
    if sorted_scores[0][1] > 0:
        return sorted_scores[0][0], "keyword-weak", sorted_scores[0][1]
    
    # 🔄 UPDATED: Default fallback
    return "legal-qa1", "default", 0


# ─── Strategy 2: Bedrock Classifier (Smart, ~$0.0003/call) ───

# 🔄 UPDATED: Adapter names in prompt
CLASSIFIER_PROMPT = """Classify this legal request into exactly ONE category.

Categories:
- "legal-qa1": General legal questions seeking explanations, definitions, or legal knowledge
- "contract-review1": Requests to review, analyze, or identify issues in a specific contract clause or agreement
- "legal-summary1": Requests to summarize a legal judgment, case, proceeding, or legal document

Request: {question}
{context_block}

Respond with ONLY the category name in quotes. Nothing else.
Example: "legal-qa1"
"""


def route_by_bedrock(question, context=""):
    """Smart LLM-based routing — ~$0.0003 per call"""
    context_block = f"\nDocument/Clause: {context[:500]}" if context else ""
    
    prompt = CLASSIFIER_PROMPT.format(
        question=question,
        context_block=context_block,
    )
    
    response = bedrock.invoke_model(
        modelId="us.anthropic.claude-3-haiku-20240307-v1:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 20,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )
    
    result = json.loads(response["body"].read())
    text = result["content"][0]["text"].strip().strip('"').strip("'")
    
    if text in VALID_ADAPTERS:
        return text, "bedrock", 1.0
    
    for name in VALID_ADAPTERS:
        if name in text.lower():
            return name, "bedrock-fuzzy", 0.8
    
    # 🔄 UPDATED: Default fallback
    return "legal-qa1", "bedrock-default", 0.5


# ─── Strategy 3: Hybrid (Best of Both) ───

def route_hybrid(question, context=""):
    """
    Keywords first (free + fast)
    Bedrock only if keywords are ambiguous (~20% of requests)
    """
    adapter, method, confidence = route_by_keywords(question, context)
    
    if method == "keyword" and confidence >= 4:
        return adapter, f"keyword(score={confidence})", confidence
    
    adapter, method, confidence = route_by_bedrock(question, context)
    return adapter, f"bedrock({method})", confidence


# ─── Invoke with Retry Logic ───

def invoke_adapter(adapter, formatted_prompt, params, max_retries=3):
    """
    All requests through Base IC + payload routing
    Enables batching across all adapters
    Retry logic handles batch queue overflow (424 errors)
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                InferenceComponentName=BASE_IC,
                Body=json.dumps({
                    "inputs": formatted_prompt,
                    "adapters": adapter,
                    "parameters": params,
                }),
                ContentType="application/json",
            )
            return response, attempt
        
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(0.3 * (attempt + 1))
                continue
            raise last_error


# ─── Main Lambda Handler ───

def lambda_handler(event, context):
    total_start = time.time()
    
    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event
    
    question = body.get("question", "")
    doc_context = body.get("context", "")
    strategy = body.get("routing_strategy", "hybrid")
    max_tokens = body.get("max_tokens", 384)
    temperature = body.get("temperature", 0.1)
    
    if not question:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No question provided"})
        }
    
    # ─── Route to correct adapter ───
    routing_start = time.time()
    
    if strategy == "keyword":
        adapter, method, confidence = route_by_keywords(question, doc_context)
    elif strategy == "bedrock":
        adapter, method, confidence = route_by_bedrock(question, doc_context)
    else:
        adapter, method, confidence = route_hybrid(question, doc_context)
    
    routing_time = time.time() - routing_start

    # ─── Format prompt ───
    if doc_context:
        full_prompt = f"{question}\n\nContext:\n{doc_context}"
    else:
        full_prompt = question
    
    formatted_prompt = f"<s>[INST] {full_prompt} [/INST]"
    
    params = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    }
    
    # ─── Invoke via Base IC + payload with retry ───
    inference_start = time.time()
    
    try:
        response, retries = invoke_adapter(adapter, formatted_prompt, params)
        
        inference_time = time.time() - inference_start
        total_time = time.time() - total_start
        
        result = json.loads(response["Body"].read().decode())
        
        if isinstance(result, list):
            generated_text = result[0].get("generated_text", "")
        else:
            generated_text = result.get("generated_text", "")
        
        if "[/INST]" in generated_text:
            generated_text = generated_text.split("[/INST]")[-1].strip()
        
        invoke_method = f"payload-batched(retries={retries})"
        routing_summary = f"Routed [{adapter}] via [{method}] invoked [{invoke_method}]"
        print(routing_summary)
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "answer": generated_text,
                "routing_summary": f"✅ [{adapter}] via [{method}] | path: {invoke_method}",
                "routing": {
                    "adapter": adapter,
                    "method": method,
                    "confidence": confidence,
                    "invoke_method": invoke_method,
                },
                "metadata": {
                    "model": "mistral-7b",
                    "endpoint": ENDPOINT_NAME,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                "latency": {
                    "routing_ms": round(routing_time * 1000, 1),
                    "inference_ms": round(inference_time * 1000, 1),
                    "total_ms": round(total_time * 1000, 1),
                },
            })
        }
    
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "routing": {
                    "adapter": adapter,
                    "method": method,
                }
            })
        }