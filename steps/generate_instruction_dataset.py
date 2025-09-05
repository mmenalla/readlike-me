import json
from pathlib import Path
from zenml import step
from typing import List, Dict
import time

from utils.openai_utils import ask_llm_generate_instruction

@step
def generate_instruction_dataset(
    chunked_passages: List[Dict],
    output_file: str = "role_based_dataset.jsonl"
) -> str:
    """
    Generates role-based instruction dataset from chunked passages.
    Each chunk gets a short instruction using OpenAI, saved as JSONL.
    """
    output_path = Path(output_file)
    records = []

    for book in chunked_passages:
        author = book.get("author", "UNKNOWN")
        chunks = book.get("chunks", [])

        for chunk in chunks:
            # Generate instruction from chunk
            try:
                instruction = ask_llm_generate_instruction(chunk)
                time.sleep(1)
            except Exception as e:
                print(f"Failed to generate instruction for chunk: {e}")
                continue

            record = {
                "role": author.split(",")[0].strip() if "," in author else author,
                "input": instruction,
                "output": chunk
            }
            records.append(record)

        with output_path.open("a", encoding="utf-8") as f:
            for rec in records:
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")

    print(f"Saved role-based dataset to {output_path}")
    return str(output_path)
