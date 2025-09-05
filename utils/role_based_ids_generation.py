import json
from pathlib import Path

input_path = Path("rb_instruction_dataset.jsonl")
output_path = input_path.with_stem(input_path.stem + "_role_based")

role_based_data = []

with input_path.open("r", encoding="utf-8") as infile:
    for line in infile:
        record = json.loads(line)
        author = record.get("author")        # full author name
        title = record.get("title", "")      # optional
        messages = record.get("messages", [])

        # Process consecutive user -> assistant messages
        for i in range(len(messages) - 1):
            if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                input_text = messages[i]["content"].strip()
                output_text = messages[i + 1]["content"].strip()

                # Create role-based instruction entry
                role_based_data.append({
                    "role": author,
                    "instruction": input_text if input_text else f"Write in the style of {author}.",
                    "output": output_text
                })
                break  # only take the first user->assistant pair per record

# Save as JSONL in role-based format
with output_path.open("w", encoding="utf-8") as outfile:
    for item in role_based_data:
        json.dump(item, outfile, ensure_ascii=False)
        outfile.write("\n")

print(f"Saved role-based dataset to {output_path.name}")
