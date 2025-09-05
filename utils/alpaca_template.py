import json
from pathlib import Path

input_path = Path("instruction_dataset.jsonl")
output_path = input_path.with_stem(input_path.stem + "_alpaca")

alpaca_data = []

with input_path.open("r", encoding="utf-8") as infile:
    for line in infile:
        record = json.loads(line)
        author = record.get("author")
        title = record.get("title")
        messages = record.get("messages", [])

        for i in range(len(messages) - 1):
            if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                instruction = f"Mimic the writing style of {author.split(', ')[1]} {author.split(', ')[0]}."
                input_text = messages[i]["content"].strip()
                output_text = messages[i + 1]["content"].strip()
                alpaca_data.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text
                })
                break

# Save as JSONL in Alpaca format
with output_path.open("w", encoding="utf-8") as outfile:
    for item in alpaca_data:
        json.dump(item, outfile, ensure_ascii=False)
        outfile.write("\n")

print(output_path.name)
