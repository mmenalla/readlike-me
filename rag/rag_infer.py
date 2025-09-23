import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rag.embed import get_embedding
from annotate import annotate_row
from transformers import AutoTokenizer, pipeline

# --- CONFIG ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "annotated_paragraphs_1"
TOP_K = 5
FINETUNED_MODEL_DIR = "./lora_finetuned_authors"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def search_prompt(prompt: str, author: str, top_k: int = TOP_K):
    """
    Annotate a prompt, embed it, and search Qdrant for the most similar chunks.
    Filters:
        - Author must match
        - At least one topic OR mood OR style_tag must match
        - Falls back to author-only if fewer than top_k results
    """
    annotation = annotate_row(author, prompt)
    if not annotation:
        print("Failed to annotate prompt")
        return [], {}

    print(f"Annotation for prompt: {annotation}")

    query_vector = get_embedding(prompt)

    # Build filters
    must_author = FieldCondition(key="author", match=MatchValue(value=author))

    should_conditions = []
    for field in ["topics", "moods", "style_tags"]:
        for val in annotation.get(field, []):
            should_conditions.append(FieldCondition(key=field, match=MatchValue(value=val)))

    combined_filter = Filter(
        must=must_author,
        should=should_conditions if should_conditions else None
    )

    client = QdrantClient(url=QDRANT_URL)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=True,
        query_filter=combined_filter
    )

    points = response.points

    if len(points) < top_k:
        print("Fewer than top_k results. Falling back to author-only filter.")
        fallback_filter = Filter(must=[must_author])
        response_author_only = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=True,
            query_filter=fallback_filter
        )
        points = response_author_only.points

    results = []
    for pt in points:
        results.append({
            "id": pt.id,
            "score": pt.score,
            "payload": pt.payload,
            "vector": pt.vector
        })
    return results, annotation


def build_generation_prompt(user_prompt: str, author: str, results: list, annotation: dict):
    """
    Build the final LLM generation prompt from search results and annotation.
    """
    context_lines = []
    for idx, pt in enumerate(results, start=1):
        payload = pt["payload"]
        paragraph = payload.get("paragraph", "")
        topics = payload.get("topics", [])
        moods = payload.get("moods", [])
        style_tags = payload.get("style_tags", [])
        context_lines.append(
            f"{idx}. {author}, topic={topics}, mood={moods}, style_tags={style_tags}\n"
            f"   Example: \"{paragraph}\""
        )

    context_str = "\n\n".join(context_lines)

    return f"""User request: {user_prompt} Mimic the style of {author}.

Context from similar {author} writings:
{context_str}

Instructions:
- Write a new paragraph in the style of {author}.
- Topic: {annotation.get("topics", [])}
- Mood: {annotation.get("moods", [])}
- Style: {annotation.get("style_tags", [])}
- Do not include explanations, meta text, or anything except the new paragraph.
- Don't include special story characters, just mimic the style of the examples.
- IMPORTANT: Return exactly ONE single paragraph, no more, no less.
"""


def run_inference(final_prompt: str):
    """
    Run inference with the LoRA fine-tuned model using the generated prompt.
    """
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_DIR)
    pipe = pipeline(
        "text-generation",
        model=FINETUNED_MODEL_DIR,
        tokenizer=tokenizer,
        device=0 if DEVICE == "mps" else -1
    )

    generated = pipe(
        final_prompt,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.2
    )

    full_text = generated[0]["generated_text"]
    new_text = full_text[len(final_prompt):].strip()

    print("=== GENERATED TEXT ===")
    print(new_text)

if __name__ == "__main__":
    prompt_text = "Describe rainy day."
    author_name = "Dostoevsky"

    results, annotation = search_prompt(prompt_text, author_name)
    final_prompt = build_generation_prompt(prompt_text, author_name, results, annotation)

    print("\n===== FINAL GENERATION PROMPT =====\n")
    print(final_prompt)

    print("\n===== MODEL OUTPUT =====\n")
    run_inference(final_prompt)
