#!/usr/bin/env python3
import os
import json
import time
import random
from datasets import load_dataset, Dataset
import openai
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
HF_TOKEN = os.getenv("HF_TOKEN")

HF_DATASET = "megi/author-style-role-balanced-with-generic"
SPLIT = "rag"
LOCAL_JSON_FILE = "annotated_author_style_rag.jsonl"
HF_NEW_DATASET_REPO = "megi/author-style-rag-annotated"

# --- Metadata config per author ---
AUTHOR_CONFIG = {
    "Kafka": {
        "topics": ["bureaucracy", "absurdity", "isolation", "dreams", "existential", "identity"],
        "moods": ["oppressive", "anxious", "surreal", "melancholic"],
        "style_tags": ["long sentences", "absurdist", "existential", "fragmented", "detached perspective", "dark", "symbolic"]
    },
    "Dostoyevsky": {
        "topics": ["crime", "guilt", "morality", "relationships", "introspection", "society", "family", "death"],
        "moods": ["anxious", "tragic", "dark", "contemplative", "melancholic"],
        "style_tags": ["long sentences", "internal monologue", "psychological", "realist", "verbose", "existential", "symbolic"]
    },

    "Tolstoy": {
        "topics": ["family", "war", "love", "society", "countryside", "morality", "relationships", "memory"],
        "moods": ["nostalgic", "contemplative", "serene", "hopeful", "romantic"],
        "style_tags": ["realist", "flowing", "metaphor-heavy", "direct narration", "verbose", "moralistic", "long sentences"]
    },
    "Hesse": {
        "topics": ["introspection", "identity", "dreams", "consciousness", "spirituality", "solitude", "nature", "existential"],
        "moods": ["contemplative", "melancholic", "hopeful", "surreal", "nostalgic", "serene"],
        "style_tags": ["lyrical", "symbolic", "stream of consciousness", "surrealist", "flowing", "metaphor-heavy", "experimental"]
    },
    "GENERIC": {
        "topics": ["weather", "city", "family", "work", "travel", "daily life", "relationships", "technology"],
        "moods": ["neutral", "serene", "joyful", "detached", "straightforward"],
        "style_tags": ["plain", "short sentences", "direct narration", "sparse detail"]
    }
}

# --- Build OpenAI prompt ---
def build_prompt(author, paragraph):
    cfg = AUTHOR_CONFIG.get(author, AUTHOR_CONFIG["GENERIC"])

    # shuffle lists
    topics = cfg['topics'][:]
    moods = cfg['moods'][:]
    style_tags = cfg['style_tags'][:]
    random.shuffle(topics)
    random.shuffle(moods)
    random.shuffle(style_tags)

    prompt =  f"""
You are a literary analyst.

Task: For the given paragraph written in a specific author's style, assign metadata fields: topic, mood, and style_tags.
Use ONLY the allowed values for the given author role.

**Pick randomly** from the allowed values.

Role: {author}
Paragraph: "{paragraph}"

Allowed metadata for this role:
- topics: {topics}
- moods: {moods}
- style_tags: {style_tags}

Requirements:
1. Pick **1 or 2 topics** from the allowed topics.
2. Pick **1 or 2 moods** from the allowed moods.
3. Pick **1 to 3 style_tags** from the allowed style_tags.
4. Return output strictly in JSON with this structure:

{{
  "author": "{author}",
  "topics": ["<TOPIC_1>", "<TOPIC_2>"],
  "moods": ["<MOOD_1>", "<MOOD_2>"],
  "style_tags": ["<STYLE_TAG_1>", "<STYLE_TAG_2>", "..."]
}}

Do not add any extra text or explanation. Only output valid JSON.
""".strip()
    print(prompt)
    return prompt

# --- OpenAI annotation call ---
def annotate_row(author, paragraph, retries=3, base_sleep=2):
    for attempt in range(retries):
        try:
            resp = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": build_prompt(author, paragraph)}],
                temperature=0.7
            )
            content = resp.choices[0].message.content.strip()
            print(f">>>>> RESPONSE:\n{content}\n")
            return json.loads(content)
        except Exception as e:
            sleep_time = base_sleep * (attempt + 1)  # incremental backoff
            print(f"Error (attempt {attempt + 1}): {e}. Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)
    return None


# --- MAIN PROCESS ---
def main():
    ds = load_dataset(HF_DATASET, split=SPLIT)
    annotated = []

    for i, row in enumerate(ds):
        author = row["role"]
        paragraph = row["output"]
        print(f"[{i}] Processing author={author}")

        metadata = annotate_row(author, paragraph)
        if metadata:
            annotated.append({
                "author": metadata["author"],
                "paragraph": paragraph,
                "topics": metadata["topics"],
                "moods": metadata["moods"],
                "style_tags": metadata["style_tags"]
            })
            print("*"*100)
        else:
            print(f"Skipping row {i}, could not annotate.")

        # Save every 50 records
        if len(annotated) % 50 == 0 and annotated:
            with open(LOCAL_JSON_FILE, "a", encoding="utf-8") as f:
                for item in annotated[-50:]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Appended {len(annotated[-50:])} records to {LOCAL_JSON_FILE}")

    # save remaining
    remaining = len(annotated) % 50
    if remaining:
        with open(LOCAL_JSON_FILE, "a", encoding="utf-8") as f:
            for item in annotated[-remaining:]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Appended remaining {remaining} records to {LOCAL_JSON_FILE}")

    # push to HF
    annotated_ds = Dataset.from_list(annotated)
    annotated_ds.push_to_hub(HF_NEW_DATASET_REPO, token=HF_TOKEN)
    print(f"Uploaded annotated dataset to HF Hub: {HF_NEW_DATASET_REPO}")


if __name__ == "__main__":
    main()
