import os
import json

from dotenv import load_dotenv
import time
import openai
from openai import RateLimitError, APIError

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_llm_passages(author: str, title: str, num_passages: int = 5, num_sentences: int = 10):
    prompt = f"""
You are a helpful assistant. Create {num_passages} passages from the book "{title}" by {author}.
Each passage should contain {num_sentences} sentences.
Generate the passages exactly as the book is written, maintaining the original style and tone.
Return the passages as a JSON array of strings without any labels or numbering, like:
[
  "First passage text ...",
  "Second passage text ...",
  "...",
  "Nth passage text ..."
]
Do not add 'Passage 1:' or similar prefixes.
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    content = response.choices[0].message.content
    try:
        passages = json.loads(content)
        return passages
    except json.JSONDecodeError:
        return content

def ask_llm_plain(num_passages: int = 5, num_sentences: int = 10):
    prompt = f"""
    You are a helpful assistant. Create {num_passages} passages with random plain information.
    Each passage should contain {num_sentences} sentences.
    Generate the passages exactly as the book is written, maintaining the original style and tone.
    Return the passages as a JSON array of strings without any labels or numbering, like:
    [
      "First passage text ...",
      "Second passage text ...",
      "...",
      "Nth passage text ..."
    ]
    Do not add 'Passage 1:' or similar prefixes.
    """
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    content = response.choices[0].message.content
    try:
        passages = json.loads(content)
        return passages
    except json.JSONDecodeError:
        return content


def ask_llm_generate_instruction(chunk: str) -> str:
    system_prompt = (
        "You are an AI assistant that reads literary passages and "
        "writes a single short instruction summarizing what the passage describes. "
        "Make it concise, in imperative form, e.g., 'Describe a rainy day in a city.'"
    )

    user_message = (
        f"Here is a literary passage:\n\n\"{chunk}\"\n\n"
        "Write a short instruction summarizing what this passage is about. "
        "Keep it under 12 words if possible."
    )

    max_retries = 5
    delay = 5  # start with 1 second

    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error: {e}")
            print(f"Rate limited or server error, retrying in {delay}s... ({attempt+1}/{max_retries})")
            time.sleep(delay)
            delay *= 2  # exponential backoff

    raise RuntimeError("Failed to generate instruction after several retries.")
