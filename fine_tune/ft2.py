#!/usr/bin/env python3
# finetune_authors_lora.py

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# --- CONFIG ---
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # smaller model recommended for testing
FINETUNED_MODEL_DIR = "./lora_finetuned_authors"
BATCH_SIZE = 1
EPOCHS = 1
LR = 3e-4
MAX_LENGTH = 512
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# --- DATA ---
DATA_FILE = "fine_tune_metallama.jsonl"

# Load dataset
ds = load_dataset("json", data_files=DATA_FILE)

# Take only the first 10 records for testing
dataset = ds["train"]
print(f"Original dataset size: {len(dataset)}")

# --- TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # LLaMA models don't have pad token

def tokenize_fn(example):
    # Concatenate instruction + input + output
    full_text = f"Instruction: {example['instruction']}\nInput: {example.get('input','')}\nOutput: {example['output']}"
    return tokenizer(full_text, truncation=True, max_length=MAX_LENGTH)

tokenized_dataset = dataset.map(tokenize_fn, batched=False)

# --- MODEL ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

# --- LoRA CONFIG ---
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, lora_config)

# --- DATA COLLATOR ---
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# --- TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=FINETUNED_MODEL_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=1,
    fp16=False,
    bf16=False,
    gradient_accumulation_steps=4,
    report_to="none"
)

# --- TRAIN ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()
trainer.save_model(FINETUNED_MODEL_DIR)

print(f"LoRA fine-tuned model saved at {FINETUNED_MODEL_DIR}")

# --- INFERENCE TEST ---
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=FINETUNED_MODEL_DIR,
    tokenizer=tokenizer,
    device=0 if DEVICE=="mps" else -1
)

prompt = "Describe a man in torment. Mimic the style of Kafka.\nInput:\nOutput:"
generated = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)

print("=== GENERATED TEXT ===")
print(generated[0]["generated_text"])
