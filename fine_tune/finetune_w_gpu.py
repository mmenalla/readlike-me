import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# --- CONFIG ---
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
FINETUNED_MODEL_DIR = "./lora_finetuned_authors_gpu"
DATA_FILE = "fine_tune_metallama.jsonl"

BATCH_SIZE = 2              # adjust based on GPU memory
EPOCHS = 10
LR = 2e-4
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- DATA ---
ds = load_dataset("json", data_files=DATA_FILE)
dataset = ds["train"]
print(f"Original dataset size: {len(dataset)}")

# --- TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(example):
    full_text = (
        f"Instruction: {example['instruction']}\n"
        f"Input: {example.get('input','')}\n"
        f"Output: {example['output']}"
    )
    return tokenizer(full_text, truncation=True, max_length=MAX_LENGTH)

tokenized_dataset = dataset.map(tokenize_fn, batched=False)

# --- MODEL ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
)

# --- LoRA CONFIG ---
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],  # typical for LLaMA
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # sanity check

# --- DATA COLLATOR (adds labels automatically) ---
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# --- TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=FINETUNED_MODEL_DIR,
    per_device_train_batch_size=1, #BATCH_SIZE,
    gradient_accumulation_steps=16,  # simulate bigger batch
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    report_to="none",
)

# --- TRAIN ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(FINETUNED_MODEL_DIR)

print(f"✅ LoRA fine-tuned model saved at {FINETUNED_MODEL_DIR}")

# --- INFERENCE TEST ---
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=FINETUNED_MODEL_DIR,
    tokenizer=tokenizer,
    device=0 if DEVICE == "cuda" else -1,
)

prompt = "Describe a man in torment. Mimic the style of Kafka.\nInput:\nOutput:"
generated = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)

print("=== GENERATED TEXT ===")
print(generated[0]["generated_text"])