import os
import json
import openai
from pymongo import MongoClient
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "rag_books"
COLLECTION_NAME = "books"
OUTPUT_FILE = "instruction_dataset.jsonl"

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

def generate_prompt_from_chunk(chunk: str) -> str:
    system_prompt = "You simplify literary text into plain, simple user instructions."
    user_message = f"Here is a literary passage:\n\n\"{chunk}\"\n\nWrite the same thing but in a simple palin way, shortly." \

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
    for record in tqdm(collection.find()):
        author = record.get("author", "")
        title = record.get("title", "")
        chunks = record.get("chunks", [])

        for chunk in chunks:
            try:
                user_prompt = generate_prompt_from_chunk(chunk)

                record_out = {
                    "author": author,
                    "title": title,
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": chunk}
                    ]
                }

                f_out.write(json.dumps(record_out, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Error with chunk: {chunk[:60]}... → {e}")
