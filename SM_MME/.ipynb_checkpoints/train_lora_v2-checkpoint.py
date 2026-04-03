%%writefile train_lora_v2.py

import os
import json
import tarfile
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
is_main = local_rank == 0

model_channel = os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model")
train_channel = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

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
            if model_path is None and os.path.exists(os.path.join(extract_path, "config.json")):
                model_path = extract_path

if is_main:
    print(f"Model: {model_path}")
    print(f"GPUs: {world_size}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    device_map={"": local_rank},
    torch_dtype=torch.bfloat16,
)

if is_main:
    print(f"Model: {model.num_parameters():,} params")

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
))
if is_main:
    model.print_trainable_parameters()

train_file = None
for f in os.listdir(train_channel):
    if f.endswith((".json", ".jsonl")):
        train_file = os.path.join(train_channel, f)
        break

dataset = load_dataset("json", data_files=train_file, split="train")
if is_main:
    print(f"Loaded {len(dataset)} examples")

def tokenize(example):
    instruction = example.get("instruction", "")
    context = example.get("context", "")
    response = example.get("output", example.get("response", ""))
    if context:
        user_msg = f"{instruction}\n\nContext:\n{context}"
    else:
        user_msg = instruction
    text = f"<s>[INST] {user_msg} [/INST] {response}</s>"
    # ─── FIX: pad to max_length=512 so all sequences are same length ───
    tokens = tokenizer(text, truncation=True, max_length=512, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

if is_main:
    print(f"Tokenized: {len(tokenized)} examples")
    effective_batch = 8 * world_size
    steps_per_epoch = len(tokenized) // effective_batch
    total_steps = steps_per_epoch * 3
    print(f"Effective batch: {effective_batch}")
    print(f"Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="/tmp/checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=8, #8 samples per GPU per step (forward+backward)
        gradient_accumulation_steps=1, #since we already have 4 GPUs!
        learning_rate=1.5e-4,#safe for QLORA
        warmup_ratio=0.03, #small learning rate for gentle updates
        lr_scheduler_type="cosine", #Slow start + slow finish, aggressive in the middle.
        bf16=True, #faster training
        tf32=True, #optimizing performance, tensor flow 32 for fast learning
        gradient_checkpointing=True, #save only a few checkpoints for backward pass
        dataloader_num_workers=4, #4 CPU processes load data in parallel so GPU never waits
        dataloader_pin_memory=True, # Pin data in CPU RAM for faster CPU→GPU transfer
        ddp_find_unused_parameters=False, ## Skip scanning for unused params each step (faster DDP)
        logging_steps=25, # Print loss/metrics every 25 steps
        logging_first_step=True, # Also log step 1 (sanity check that training started OK)
        save_strategy="no", #no checkpoints saved during training
        report_to="wandb",
        run_name="legal-qa-v2-512-r32-3ep",
        optim="paged_adamw_8bit", #safety net if it runs out of memory, saves memory, best optimizer for fine tuning, 
    ),
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), #batches data for GPU, mask language modeling, GPT style
)

result = trainer.train()

if is_main:
    print(f"Loss: {result.metrics.get('train_loss', 'N/A')}")
    print(f"Runtime: {result.metrics.get('train_runtime', 0)/60:.1f} min")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {size:>10,} bytes  {f}")
    print("DONE!")
