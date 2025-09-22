from datasets import load_dataset
import json

HF_DS_NAME = "megi/author-style-role-balanced-with-generic"
SPLIT = "fine_tune"
OUTPUT_FILE = "fine_tune_metallama.jsonl"

# load the fine-tune split
ds = load_dataset(HF_DS_NAME, split=SPLIT)

print("Columns in dataset:", ds.column_names)  # should include ['role', 'input', 'output']

# convert each record to MetaLLaMA format
def convert_to_metallama_format(example):
    return {
        "instruction": f"{example['input'].strip()} Mimic the style of {example['role']}",
        "input": "",
        "output": example['output'].strip()
    }

ft_ds = ds.map(convert_to_metallama_format, remove_columns=["role", "text", "input"])
ft_ds = ft_ds.select_columns(["instruction", "input", "output"])

print(ft_ds.column_names)
print(f"Total records in fine-tune dataset: {len(ft_ds)}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for record in ft_ds:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Fine-tuning dataset ready: {len(ft_ds)} records")
print(f"Saved to {OUTPUT_FILE}")
