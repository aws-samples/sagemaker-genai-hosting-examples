
import os
import json
import torch
import tarfile
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_channel = os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model")
adapter_channel = os.environ.get("SM_CHANNEL_ADAPTER", "/opt/ml/input/data/adapter")
test_channel = os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test")
output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

# ─── Find base model ───
model_path = None
if os.path.exists(os.path.join(model_channel, "config.json")):
    model_path = model_channel
else:
    for root, dirs, files in os.walk(model_channel):
        if "config.json" in files:
            model_path = root
            break
if model_path is None:
    for f in os.listdir(model_channel):
        if f.endswith(".tar.gz"):
            extract_path = "/tmp/model"
            os.makedirs(extract_path, exist_ok=True)
            with tarfile.open(os.path.join(model_channel, f), "r:gz") as tar:
                tar.extractall(extract_path)
            for root, dirs, files in os.walk(extract_path):
                if "config.json" in files:
                    model_path = root
                    break

print(f"Base model: {model_path}")

# ─── Find adapter ───
adapter_path = None
if os.path.exists(os.path.join(adapter_channel, "adapter_config.json")):
    adapter_path = adapter_channel
else:
    for root, dirs, files in os.walk(adapter_channel):
        if "adapter_config.json" in files:
            adapter_path = root
            break
if adapter_path is None:
    for f in os.listdir(adapter_channel):
        if f.endswith(".tar.gz"):
            extract_path = "/tmp/adapter"
            os.makedirs(extract_path, exist_ok=True)
            with tarfile.open(os.path.join(adapter_channel, f), "r:gz") as tar:
                tar.extractall(extract_path)
            for root, dirs, files in os.walk(extract_path):
                if "adapter_config.json" in files:
                    adapter_path = root
                    break

print(f"Adapter: {adapter_path}")
print(f"Adapter files: {os.listdir(adapter_path)}")

# ─── Load test set from S3 channel ───
test_file = None
for f in os.listdir(test_channel):
    if f.endswith(".json"):
        test_file = os.path.join(test_channel, f)
        break

print(f"Test file: {test_file}")

with open(test_file, "r") as fh:
    test_set = json.load(fh)

print(f"Test set: {len(test_set)} examples")

# ─── Load model + adapter ───
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

print("Loading adapter...")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()
print(f"Model loaded: {model.num_parameters():,} params")


def generate_response(instruction, context="", max_new_tokens=512):
    if context:
        user_msg = instruction + "\n\nContext:\n" + context
    else:
        user_msg = instruction

    prompt = "<s>[INST] " + user_msg + " [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return generated.strip()


print("=" * 60)
print(f"Running inference on {len(test_set)} test examples...")
print("=" * 60)

results = []
for i, ex in enumerate(test_set):
    instruction = ex.get("instruction", "")
    context = ex.get("context", "")

    pred = generate_response(instruction, context)

    result = {
        "id": ex.get("id", f"q_{i}"),
        "category": ex.get("category", "unknown"),
        "difficulty": ex.get("difficulty", "unknown"),
        "instruction": instruction,
        "context": context,
        "expected_output": ex.get("expected_output", ""),
        "key_facts": ex.get("key_facts", []),
        "predicted": pred,
    }
    results.append(result)

    print(f"[{i+1}/{len(test_set)}] {result['id']} | {result['difficulty']}")
    print(f"  Q: {instruction[:80]}...")
    print(f"  A: {pred[:150]}...")
    print()

output_file = os.path.join(output_dir, "eval_predictions.json")
with open(output_file, "w") as fh:
    json.dump(results, fh, indent=2)

print(f"Saved {len(results)} predictions to {output_file}")
print("DONE!")
