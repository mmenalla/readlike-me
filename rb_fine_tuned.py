from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

# --------------------------------------------------
# Config
# --------------------------------------------------
# MODEL_NAME = "meta-llama/Llama-3.1-8B"
# MODEL_NAME = "meta-llama/Llama-2-3B-hf"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

DATASET_NAME = "megi/author-style-role-based-dataset"
OUTPUT_DIR = "./llama-finetuned"

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
dataset = load_dataset(DATASET_NAME)

# Take only 1% of the dataset for training/validation
train_frac = 0.01
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(int(len(dataset["train"]) * train_frac)))
if "validation" in dataset:
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(int(len(dataset["validation"]) * train_frac)))

# --------------------------------------------------
# Load tokenizer and model
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # LLaMA models often need explicit pad_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16  # or float16 if bfloat16 not available
)

# --------------------------------------------------
# Tokenization
# --------------------------------------------------
def tokenize_fn(batch):
    encodings = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)

# --------------------------------------------------
# Training setup
# --------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,   # effective batch size = 16
    num_train_epochs=3,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    save_total_limit=2,
    # fp16=True,
    push_to_hub=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset.get("validation"),
    tokenizer=tokenizer,
)

# --------------------------------------------------
# Train
# --------------------------------------------------
trainer.train()

# --------------------------------------------------
# Save final model
# --------------------------------------------------
trainer.push_to_hub()
