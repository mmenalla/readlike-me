import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rag.embed import get_embedding
from annotate import annotate_row
from transformers import AutoTokenizer, pipeline

# --- CONFIG ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "vdb1"
TOP_K = 3
FINETUNED_MODEL_DIR = "./lora_finetuned_authors"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# def search_prompt(prompt: str, author: str, top_k: int = TOP_K):
#     """
#     Annotate a prompt, embed it, and search Qdrant for the most similar chunks.
#     Filters:
#         - Author must match
#         - At least one topic OR mood OR style_tag must match
#         - Falls back to author-only if fewer than top_k results
#     """
#     annotation = annotate_row(author, prompt)
#     if not annotation:
#         print("Failed to annotate prompt")
#         return [], {}
#
#     print(f"Annotation for prompt: {annotation}")
#
#     query_vector = get_embedding(prompt)
#
#     # Build filters
#     must_author = FieldCondition(key="author", match=MatchValue(value=author))
#
#     should_conditions = []
#     for field in ["topics", "moods", "style_tags"]:
#         for val in annotation.get(field, []):
#             should_conditions.append(FieldCondition(key=field, match=MatchValue(value=val)))
#
#     combined_filter = Filter(
#         must=must_author,
#         should=should_conditions if should_conditions else None
#     )
#
#     client = QdrantClient(url=QDRANT_URL)
#     response = client.query_points(
#         collection_name=COLLECTION_NAME,
#         query=query_vector,
#         limit=top_k,
#         with_payload=True,
#         with_vectors=True,
#         query_filter=combined_filter
#     )
#
#     points = response.points
#
#     if len(points) < top_k:
#         print("Fewer than top_k results. Falling back to author-only filter.")
#         fallback_filter = Filter(must=[must_author])
#         response_author_only = client.query_points(
#             collection_name=COLLECTION_NAME,
#             query=query_vector,
#             limit=top_k,
#             with_payload=True,
#             with_vectors=True,
#             query_filter=fallback_filter
#         )
#         points = response_author_only.points
#
#     results = []
#     for pt in points:
#         results.append({
#             "id": pt.id,
#             "score": pt.score,
#             "payload": pt.payload,
#             "vector": pt.vector
#         })
#     return results, annotation
def search_prompt(prompt: str, author: str, top_k: int = TOP_K):
    """
    Annotate a prompt, embed it, and search Qdrant for the most similar chunks.
    Filters:
        - Author must match
        - At least one topic OR mood OR style_tag must match
        - Falls back to author-only if fewer than top_k results
    Deduplicates paragraphs before returning.
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
        limit=top_k * 3,  # fetch more than needed, to allow dedup
        with_payload=True,
        with_vectors=True,
        query_filter=combined_filter
    )

    points = response.points

    # Fallback to author-only if fewer than top_k
    if len(points) < top_k:
        print("Fewer than top_k results. Falling back to author-only filter.")
        fallback_filter = Filter(must=[must_author])
        response_author_only = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k * 3,
            with_payload=True,
            with_vectors=True,
            query_filter=fallback_filter
        )
        points = response_author_only.points

    # ✅ Deduplicate by paragraph text
    seen_paragraphs = set()
    deduped_points = []
    for pt in points:
        para = pt.payload.get("paragraph", "")
        if para not in seen_paragraphs:
            seen_paragraphs.add(para)
            deduped_points.append(pt)
        if len(deduped_points) >= top_k:
            break

    results = []
    for pt in deduped_points:
        results.append({
            "id": pt.id,
            "score": pt.score,
            "payload": pt.payload,
            "vector": pt.vector
        })

    # return both
    return results, annotation


# def build_generation_prompt(user_prompt: str, author: str, results: list, annotation: dict):
#     """
#     Build the final LLM generation prompt from search results and annotation.
#     """
#     context_lines = []
#     for idx, pt in enumerate(results, start=1):
#         payload = pt["payload"]
#         paragraph = payload.get("paragraph", "")
#         topics = payload.get("topics", [])
#         moods = payload.get("moods", [])
#         style_tags = payload.get("style_tags", [])
#         context_lines.append(
#             f"{idx}. {author}, topic={topics}, mood={moods}, style_tags={style_tags}\n"
#             f"   Example: \"{paragraph}\""
#         )
#
#     context_str = "\n\n".join(context_lines)
#
#     return f"""{user_prompt} Mimic the style of {author}.
#
# Context from similar {author} writings:
# {context_str}
#
# Instructions:
# - Write 1-3 sentences in the style of {author} about {user_prompt}.
# - Topic: {annotation.get("topics", [])}
# - Mood: {annotation.get("moods", [])}
# - Style: {annotation.get("style_tags", [])}
# - Do not include explanations, meta text, or anything except the new text.
# - Don't include special story characters, names or stories, just mimic the style of the examples.
# """
def build_generation_prompt(user_prompt: str, author: str, results: list, annotation: dict):
    """
    Build the final LLM generation prompt from search results and annotation.
    Only the first 2 sentences of each example paragraph are included.
    """
    context_lines = []
    for idx, pt in enumerate(results, start=1):
        payload = pt["payload"]
        paragraph = payload.get("paragraph", "")
        # Split into sentences (simple heuristic: split by period)
        sentences = paragraph.split(". ")
        first_two = ". ".join(sentences[:2]).strip()
        if not first_two.endswith("."):
            first_two += "."
        topics = payload.get("topics", [])
        moods = payload.get("moods", [])
        style_tags = payload.get("style_tags", [])
        context_lines.append(
            f"{idx}. {author}, topic={topics}, mood={moods}, style_tags={style_tags}\n"
            f"   Example: \"{first_two}\""
        )

    context_str = "\n\n".join(context_lines)

    return f"""{user_prompt} Mimic the style of {author}.

Context from similar {author} writings:
{context_str}

Instructions:
- Write 1-3 sentences in the style of {author} about {user_prompt}.
- Topic: {annotation.get("topics", [])}
- Mood: {annotation.get("moods", [])}
- Style: {annotation.get("style_tags", [])}
- Do not include explanations, meta text, or anything except the new text.
- Don't include special story characters, names or stories, just mimic the style of the examples.
"""

# def build_generation_prompt(user_prompt: str, author: str, results: list, annotation: dict):
#     """
#     Build the final LLM generation prompt from search results and annotation.
#     Deduplicates and anonymizes example paragraphs to avoid copying specific characters.
#     """
#     style_examples = []
#     for pt in results:
#         payload = pt["payload"]
#         paragraph = payload.get("paragraph", "")
#         # Remove names and characters: replace capitalized words (simple heuristic)
#         anonymized_para = " ".join(
#             "[REDACTED]" if w.istitle() else w for w in paragraph.split()
#         )
#         style_examples.append(anonymized_para)
#
#     # Take first 2-3 anonymized examples to illustrate style
#     context_str = "\n".join(f"{idx+1}. {ex}" for idx, ex in enumerate(style_examples[:3]))
#
#     return f"""User request: {user_prompt} Mimic the style of {author}.
#
# Style context from similar writings:
# {context_str}
#
# Style cues:
# - Topics: {annotation.get("topics", [])}
# - Moods: {annotation.get("moods", [])}
# - Style tags: {annotation.get("style_tags", [])}
#
# Instructions:
# - Write 1-3 sentences in the style of {author} about '{user_prompt}'.
# - Do not include explanations, meta text, or anything except the new text.
# - Do not use special story characters, names, or specific plots; focus on style only.
# """


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
        temperature=0.9,
        repetition_penalty=1.2
    )

    full_text = generated[0]["generated_text"]
    new_text = full_text[len(final_prompt):].strip()

    print("=== GENERATED TEXT ===")
    print(full_text)
from sentence_transformers import CrossEncoder

# Load a cross-encoder for semantic relevance scoring
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=0 if torch.backends.mps.is_available() else "cpu")

def rerank_chunks(query: str, chunks: list):
    """
    Rerank the retrieved chunks using a cross-encoder.
    chunks: list of dicts {"id", "payload", "vector", ...}
    Returns: chunks sorted by relevance
    """
    # Prepare pairs: (query, chunk_text)
    pairs = [(query, pt["payload"]["paragraph"]) for pt in chunks]

    # Get relevance scores
    scores = reranker.predict(pairs)

    # Sort chunks by score descending
    sorted_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
    return sorted_chunks

def run_inference_with_rag(author_name, prompt_text):
    # prompt_text = "Describe a man with a guilty consciousness."
    # author_name = "Kafka"

    results, annotation = search_prompt(prompt_text, author_name)
    # Rerank by semantic relevance
    results = rerank_chunks(prompt_text, results)
    final_prompt = build_generation_prompt(prompt_text, author_name, results, annotation)

    print("\n===== FINAL GENERATION PROMPT =====\n")
    print(final_prompt)

    print("\n===== MODEL OUTPUT =====\n")
    # run_inference("Describe a man with a guilty consciousness. Mimic the style of Dostoevsky. Just mimic the style, do not include characters or stories")
    run_inference(final_prompt)
