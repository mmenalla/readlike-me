import json
import os

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch
from sklearn.model_selection import train_test_split

HF_TOKEN = os.getenv("HF_TOKEN")
model_name = "meta-llama/Llama-3.1-8B"
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
data_path = "/Users/megi/Documents/Other/LLM/readlike-me/utils/instruction_dataset_alpaca.jsonl"
alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

# ---- Load and process the dataset ----
def load_and_format_alpaca(path):
    with open(path, "r") as f:
        records = [json.loads(line) for line in f]

    for record in records:
        input_text = alpaca_template.format(
            instruction=record["instruction"],
            input=record.get("input", ""),
            output=record["output"]
        )
        record["text"] = input_text
    return Dataset.from_list(records)

dataset = load_and_format_alpaca(data_path)
train_data, test_data = train_test_split(dataset.to_list(), test_size=0.1)
train_ds = Dataset.from_list(train_data)
eval_ds = Dataset.from_list(test_data)

# ---- Tokenizer and model ----
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # device_map="auto",
    # load_in_8bit=True,  # requires bitsandbytes
    # trust_remote_code=True
)

# ---- Apply LoRA ----
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# ---- Tokenization ----
def tokenize(example):
    tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_ds = train_ds.map(tokenize, batched=True)
eval_ds = eval_ds.map(tokenize, batched=True)

# ---- Trainer setup ----
training_args = TrainingArguments(
    output_dir="./lora-alpaca-checkpoints",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="steps",
    logging_steps=20,
    eval_steps=100,
    save_steps=200,
    num_train_epochs=1,
    learning_rate=2e-4,
    save_total_limit=2,
    fp16=False,
    report_to="none",
    no_cuda=True
)

data_collator = None #DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ---- Train ----
trainer.train()

# ---- Save final LoRA model ----
model.save_pretrained("./lora-alpaca-final")
tokenizer.save_pretrained("./lora-alpaca-final")
