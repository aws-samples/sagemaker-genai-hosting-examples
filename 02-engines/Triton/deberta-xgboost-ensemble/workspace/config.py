"""
Configuration and constants for Triton NLI DeBERTa + XGBoost ensemble deployment.

Contains:
- NLI model settings and candidate labels
- Triton model configuration generators (config.pbtxt files)
- Benchmark sample texts

Architecture:
  text → [NLI(text, label_1), ..., NLI(text, label_N)] → confidence_vector(1, N) → XGBoost
"""

# ---------------------------------------------------------------------------
# Model & Label Constants
# ---------------------------------------------------------------------------

NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-base"
N_LABELS = 50
MAX_SEQ_LEN = 128

# Matches HuggingFace ZeroShotClassificationPipeline hypothesis format
HYPOTHESIS_TEMPLATE = "This example is {}."

# Label indices for cross-encoder/nli-deberta-v3-base
# config.json: {0: "contradiction", 1: "entailment", 2: "neutral"}
ENTAILMENT_IDX = 1
CONTRADICTION_IDX = 0


# Default label set — an email threat-detection taxonomy. The NLI cross-encoder
# scores each email body against every threat category, and a downstream
# XGBoost head converts the resulting confidence vector into a malicious-or-
# benign decision.
# Swap this for your own domain (support-ticket intents, moderation policies,
# compliance flags, etc.) — length must match N_LABELS above.
NLI_LABELS = [
    "phishing",
    "credential harvesting",
    "business email compromise",
    "invoice fraud",
    "gift-card scam",
    "wire-transfer fraud",
    "payroll redirection fraud",
    "vendor impersonation",
    "executive impersonation",
    "spoofed sender",
    "lookalike domain",
    "reply-chain hijack",
    "malware attachment",
    "malicious link",
    "drive-by download",
    "ransomware delivery",
    "macro-enabled document",
    "password-protected archive",
    "fake login page",
    "OAuth consent phishing",
    "MFA fatigue prompt",
    "account takeover attempt",
    "bank credential theft",
    "cryptocurrency theft",
    "tax refund scam",
    "tech support scam",
    "extortion",
    "advance-fee fraud",
    "inheritance scam",
    "lottery scam",
    "romance scam",
    "job offer scam",
    "recruitment scam",
    "charity fraud",
    "fake invoice",
    "purchase order fraud",
    "shipping notification spoof",
    "delivery failure lure",
    "fake voicemail notification",
    "calendar-invite phishing",
    "document-share lure",
    "IT helpdesk impersonation",
    "HR policy impersonation",
    "benefits enrollment lure",
    "urgent action request",
    "data exfiltration attempt",
    "typosquatted link",
    "QR-code phishing",
    "callback phishing",
    "unsolicited marketing",
]

assert len(NLI_LABELS) == N_LABELS, (
    f"NLI_LABELS length ({len(NLI_LABELS)}) must match N_LABELS ({N_LABELS})"
)


# ---------------------------------------------------------------------------
# Triton Model Configuration Generators
# ---------------------------------------------------------------------------

def get_nli_config() -> str:
    """
    Generate NLI DeBERTa Triton config.

    max_batch_size=0 because the ensemble does fan-out (1 text → N NLI pairs)
    and fan-in (N scores → 1 confidence vector). Triton's ensemble scheduler
    propagates the batch dim uniformly, so we manage shapes explicitly.

    Input:  input_ids [N_LABELS, MAX_SEQ_LEN], attention_mask [N_LABELS, MAX_SEQ_LEN]
    Output: confidence_vector [1, N_LABELS]
    """
    return f"""
name: "nli_deberta"
backend: "onnxruntime"
max_batch_size: 0

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ {N_LABELS}, {MAX_SEQ_LEN} ]
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ {N_LABELS}, {MAX_SEQ_LEN} ]
  }}
]

output [
  {{
    name: "confidence_vector"
    data_type: TYPE_FP32
    dims: [ 1, {N_LABELS} ]
  }}
]

instance_group [
  {{ kind: KIND_GPU, count: 1 }}
]

optimization {{
  graph {{ level: 3 }}
  execution_accelerators {{
    gpu_execution_accelerator [
      {{
        name: "tensorrt"
        parameters {{ key: "precision_mode" value: "FP16" }}
        parameters {{ key: "max_workspace_size_bytes" value: "8589934592" }}
      }}
    ]
  }}
}}
""".strip()


def get_xgb_config() -> str:
    """
    Generate XGBoost Triton config.

    Input:  float_input [1, N_LABELS]  (the NLI confidence vector)
    Output: label [1], probabilities [1, 2]
    """
    return f"""
name: "xgboost_classifier"
backend: "onnxruntime"
max_batch_size: 0

input [
  {{
    name: "float_input"
    data_type: TYPE_FP32
    dims: [ 1, {N_LABELS} ]
  }}
]

output [
  {{
    name: "label"
    data_type: TYPE_INT64
    dims: [ 1 ]
  }},
  {{
    name: "probabilities"
    data_type: TYPE_FP32
    dims: [ 1, 2 ]
  }}
]

instance_group [
  {{ kind: KIND_CPU, count: 1 }}
]
""".strip()


def get_ensemble_config() -> str:
    """
    Generate ensemble Triton config.

    Chains: nli_deberta → xgboost_classifier
    Maps nli_deberta's confidence_vector output to xgboost's float_input.
    """
    return f"""
name: "ensemble_model"
platform: "ensemble"
max_batch_size: 0

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ {N_LABELS}, {MAX_SEQ_LEN} ]
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ {N_LABELS}, {MAX_SEQ_LEN} ]
  }}
]

output [
  {{
    name: "PREDICTION"
    data_type: TYPE_INT64
    dims: [ 1 ]
  }}
]

ensemble_scheduling {{
  step [
    {{
      model_name: "nli_deberta"
      model_version: -1
      input_map {{
        key: "input_ids"
        value: "input_ids"
      }}
      input_map {{
        key: "attention_mask"
        value: "attention_mask"
      }}
      output_map {{
        key: "confidence_vector"
        value: "confidence_vector"
      }}
    }},
    {{
      model_name: "xgboost_classifier"
      model_version: -1
      input_map {{
        key: "float_input"
        value: "confidence_vector"
      }}
      output_map {{
        key: "label"
        value: "PREDICTION"
      }}
    }}
  ]
}}
""".strip()


# ---------------------------------------------------------------------------
# Benchmark Texts
# ---------------------------------------------------------------------------
#
# Short email-body snippets that exercise the full fan-out pipeline (each text
# is scored against all N_LABELS threat categories). Mix of benign and
# suspicious content. Swap for inputs from your own domain if you want sample
# predictions to reflect it.

BENCHMARK_TEXTS = [
    "Your Office365 password expires today. Click here to verify your credentials and avoid account lockout.",
    "Hi team, please find the Q3 financial review deck attached ahead of Thursday's meeting.",
    "URGENT: I'm in a meeting and need you to pick up five $500 Apple gift cards for a client. Reimburse you tonight. — CEO",
    "Reminder: your Zoom meeting with the platform team is scheduled for 10:30 AM tomorrow.",
    "Attached is the updated invoice — please remit payment to our new bank account by end of week.",
    "Thanks for the detailed code review, I've pushed the requested changes to the PR.",
    "Your package could not be delivered. Confirm your address and pay the $1.99 redelivery fee here.",
    "FYI — onboarding docs for the new hire are in the shared drive under People Ops / 2026.",
]
