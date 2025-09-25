import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import CrossEncoder
import re

from embed import get_embedding
from annotate import annotate_row

# --- CONFIG ---
MODEL_DIR = "./rag_module/lora_finetuned_authors"
AUTHORS = {
    "Dostoevsky": "Fyodor Dostoevsky",
    "Tolstoy": "Leo Tolstoy",
    "Hesse": "Hermann Hesse",
    "Kafka": "Franz Kafka",
    "GENERIC": "Generic Author"
}
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "vdb1"
TOP_K = 3
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# --- LOAD MODEL AND TOKENIZER ONCE ---
@st.cache_resource(show_spinner=True)
def load_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=torch.float32
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model()

# --- UTILS ---
def clean_generated_text(text: str, prompt: str = None, max_sentences=5):
    """
    Remove unwanted prefixes, Input/Output labels, and truncate sentences.
    """
    if prompt and text.startswith(prompt):
        text = text[len(prompt):]
    for label in ["Input:", "Output:"]:
        text = text.replace(label, "")
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sentences[:max_sentences])

def truncate_sentences(text, max_sentences=5):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sentences[:max_sentences])

# --- RAG / Qdrant retrieval ---
def search_prompt(prompt: str, author: str, top_k: int = TOP_K):
    annotation = annotate_row(author, prompt)
    if not annotation:
        print("Failed to annotate prompt")
        return [], {}
    query_vector = get_embedding(prompt)
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
        limit=top_k * 3,
        with_payload=True,
        with_vectors=True,
        query_filter=combined_filter
    )
    points = response.points

    if len(points) < top_k:
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
    return results, annotation

def build_generation_prompt(user_prompt: str, author: str, results: list, annotation: dict, max_sentences: int = 5):
    """
    Build a final LLM generation prompt that includes RAG context examples.
    Each example paragraph is truncated to `max_sentences`.
    """
    # Build RAG context
    context_lines = []
    for idx, pt in enumerate(results, start=1):
        para = pt["payload"].get("paragraph", "")
        # Clean paragraph from Input/Output labels
        para = para.replace("Input:", "").replace("Output:", "").strip()
        # Truncate to max_sentences
        sentences = re.split(r'(?<=[.!?])\s+', para)
        truncated = " ".join(sentences[:max_sentences])
        context_lines.append(f"{idx}. {truncated}")

    rag_context = "\n".join(context_lines)

    final_prompt = f"""
    {user_prompt}Mimic the style of {author}.
    - Topics: {annotation.get('topics', [])}
    - Mood: {annotation.get('moods', [])}
    - Style tags: {annotation.get('style_tags', [])}
    - Only generate new text; do not copy or repeat phrases from the examples.
    

    Use the following example paragraphs as guidance for style:
    {rag_context}
    
    Generate in 1-3 sentences.
    """

    return final_prompt

# --- MODEL INFERENCE ---
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=0 if torch.backends.mps.is_available() else "cpu")

def rerank_chunks(query: str, chunks: list):
    pairs = [(query, pt["payload"]["paragraph"]) for pt in chunks]
    scores = reranker.predict(pairs)
    sorted_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
    return sorted_chunks

def run_inference_lora(author_name, user_prompt, max_length=256, max_sentences=5):
    prompt = f"{user_prompt} Mimic the style of {author_name}."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,  # ✅ generate up to 300 new tokens
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=3
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_generated_text(text, prompt, max_sentences)

def run_inference_rag(author_name, user_prompt, max_sentences=5):
    # Step 1: Search and rerank
    results, annotation = search_prompt(user_prompt, author_name)
    results = rerank_chunks(user_prompt, results)

    # Step 2: Build prompt for LLM
    final_prompt = build_generation_prompt(user_prompt, author_name, results, annotation)
    print(final_prompt)

    # Step 5: Run LORA inference
    text = run_inference_lora(author_name=author_name, user_prompt=final_prompt)
    # return f"{final_prompt}\n\n{text}"
    return text

# --- STREAMLIT APP ---
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("READLikeMe - Mimic literary style LLM")

selected_author_key = st.selectbox("Select author:", list(AUTHORS.keys()))
selected_author_fullname = AUTHORS[selected_author_key]

st.subheader(f"{selected_author_fullname} talking...")

user_prompt = st.text_input("Your message:", key="input_text")
use_rag = st.checkbox("Use RAG")

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Generate") and user_prompt.strip():
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        with st.spinner("Generating..."):
            if use_rag:
                response = run_inference_rag(selected_author_key, user_prompt)
            else:
                response = run_inference_lora(selected_author_key, user_prompt)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "author_key": selected_author_key
        })
with col2:
    if st.button("Clear"):
        st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.write(f"You: {msg['content']}")
    else:
        author_fullname = AUTHORS.get(msg["author_key"], "Author")
        st.write(f"{author_fullname}: {msg['content']}")
        st.markdown("---")
