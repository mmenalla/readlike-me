import requests
import random
import re
from rapidfuzz import fuzz

def is_author_match(query, candidate, threshold=80) -> bool:
    return fuzz.ratio(query.lower(), candidate.lower()) >= threshold

def get_plaintext_url(formats: dict) -> str:
    for mime, url in formats.items():
        if 'text/plain' in mime or 'text' in mime:
            return url
    for mime, url in formats.items():
        if 'htm' in mime or 'html' in url:
            return url
    for mime, url in formats.items():
        if 'pdf' in mime:
            return url
    raise ValueError("No suitable format found in formats.")

def split_into_sentences(text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def get_gutenberg_passages(author: str, title: str, count=5, sentences_per_passage=10) -> list[str]:
    search_url = "https://gutendex.com/books"
    params = {"search": title, "languages": "en"}
    r = requests.get(search_url, params=params)
    r.raise_for_status()
    data = r.json()

    # Find book by author
    for result in data["results"]:
        if is_author_match(author, result["authors"][0]["name"]):
            book = result
            break
    else:
        print(f"No match found for {author} - {title}")
        return []

    text_url = get_plaintext_url(book["formats"])
    print(f"text_url: {text_url}")
    text = requests.get(text_url).text

    # Strip Gutenberg headers/footers
    start_marker = "*** START OF"
    end_marker = "*** END OF"
    start_index = text.find(start_marker)
    end_index = text.find(end_marker)
    if start_index != -1 and end_index != -1:
        text = text[start_index + len(start_marker):end_index]

    sentences = split_into_sentences(text)
    total_sentences = len(sentences)

    if total_sentences <= sentences_per_passage:
        return [" ".join(sentences)]

    # Generate non-overlapping start indices
    possible_indices = list(range(0, total_sentences - sentences_per_passage))
    random.shuffle(possible_indices)
    selected_indices = []

    for idx in possible_indices:
        # enforce no overlap with already selected passages
        if all(abs(idx - existing) >= sentences_per_passage for existing in selected_indices):
            selected_indices.append(idx)
        if len(selected_indices) >= count:
            break

    passages = []
    for i in selected_indices:
        passage = sentences[i:i + sentences_per_passage]
        # Extend to complete sentence if previous logic is needed
        j = i + sentences_per_passage
        while j < total_sentences and not sentences[j - 1].endswith(('.', '!', '?')):
            passage.append(sentences[j])
            j += 1
        cleaned = re.sub(r'<[^>]+>', '', re.sub(r'\s+', ' ', " ".join(passage))).strip()
        passages.append(cleaned)

    return passages
