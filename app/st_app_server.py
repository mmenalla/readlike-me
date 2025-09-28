import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import CrossEncoder
import re

from annotate import annotate_row
from embed import get_embedding

# --- CONFIG ---
MODEL_DIR = "./lora_finetuned_authors_gpu"
BASE_MODEL = "meta-llama/Llama-2-7b-hf"
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
DEVICE = 0 if torch.cuda.is_available() else -1  # Hugging Face pipeline expects int

# --- LOAD PIPELINES ONCE ---
@st.cache_resource(show_spinner=True)
def load_pipelines():
    lora_pipe = pipeline(
        "text-generation",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        device=DEVICE,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    base_pipe = pipeline(
        "text-generation",
        model=BASE_MODEL,
        tokenizer=BASE_MODEL,
        device=DEVICE,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device=DEVICE if DEVICE >= 0 else "cpu"
    )
    return lora_pipe, base_pipe, reranker

lora_pipe, base_pipe, reranker = load_pipelines()

# --- UTILS ---
#def suffix_prompt(prompt: str, author: str):
#    return f"{prompt} Mimic the style of {author}.\nInput:\nOutput:"
def suffix_prompt(prompt: str, author: str):
    enhanced_prompt =  (
        f"{prompt}\n\n"
        f"Write in the literary style of {author}. "
        f"Do not invent characters, names, or dialogues. "
        f"Do not describe specific events or fictional cases. "
        f"Instead, provide a rich, reflective, and descriptive commentary "
        f"that captures the essence of the request with depth and subtlety.\n\n"
        f"Answer:\n"
    )
    print(f"New prompt: {enhanced_prompt}")
    return enhanced_prompt

def clean_generated_text(text: str, prompt: str = None, max_sentences=5):
    if prompt and text.startswith(prompt):
        text = text[len(prompt):]
    text = text.strip()
    sentences = re.split(
        r'(?<=[.!?])\s+', text.split("Output: ", 1)[-1]
    ) if "Output: " in text else re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sentences[:max_sentences])

# --- RAG / Qdrant retrieval ---
def search_prompt(prompt: str, author: str, top_k: int = TOP_K):
    annotation = annotate_row(author, prompt)
    if not annotation:
        return [], {}
    query_vector = get_embedding(prompt)
    must_author = FieldCondition(key="author", match=MatchValue(value=author))

    should_conditions = []
    for field in ["topics", "moods", "style_tags"]:
        for val in annotation.get(field, []):
            should_conditions.append(FieldCondition(key=field, match=MatchValue(value=val)))

    combined_filter = Filter(must=must_author, should=should_conditions if should_conditions else None)
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

    seen_paragraphs, deduped_points = set(), []
    for pt in points:
        para = pt.payload.get("paragraph", "")
        if para not in seen_paragraphs:
            seen_paragraphs.add(para)
            deduped_points.append(pt)

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
    context_lines = []
    for idx, pt in enumerate(results, start=1):
        para = pt["payload"].get("paragraph", "")
        para = para.replace("Input:", "").replace("Output:", "").strip()
        sentences = re.split(r'(?<=[.!?])\s+', para)
        truncated = " ".join(sentences[:max_sentences])
        context_lines.append(f"{idx}. {truncated}")

    rag_context = "\n".join(context_lines)
    final_prompt = f"""
{user_prompt} \n\n Mimic the literary style of {author}.
- Do not invent characters, names, or dialogues.
- Do not describe specific events or fictional cases. 
- Instead, provide a rich, reflective, and descriptive commentary that captures the essence of the request with depth and subtlety.\n

- Topics: {", ".join(annotation.get('topics', []))}
- Mood: {", ".join(annotation.get('moods', []))}
- Style tags: {", ".join(annotation.get('style_tags', []))}

Use the following example paragraphs as guidance for style:
{rag_context}

Generate in 1-3 sentences.\n\n
Answer:
"""
    print(f"RAG Prompt: {final_prompt}")
    return final_prompt

def rerank_chunks(query: str, chunks: list):
    pairs = [(query, pt["payload"]["paragraph"]) for pt in chunks]
    scores = reranker.predict(pairs)
    sorted_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
    return sorted_chunks[:TOP_K]

# --- INFERENCE ---
def run_inference(pipe, prompt, max_new_tokens, temperature, top_p, repetition_penalty, top_k, max_sentences=5):
    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
    )
    text = outputs[0]["generated_text"]
    return clean_generated_text(text, prompt, max_sentences)

def run_inference_lora(author_name, user_prompt, **gen_cfg):
    prompt = suffix_prompt(user_prompt, author_name)
    return run_inference(lora_pipe, prompt, **gen_cfg)

def run_inference_base(author_name, user_prompt, **gen_cfg):
    prompt = suffix_prompt(user_prompt, author_name)
    return run_inference(base_pipe, prompt, **gen_cfg)

def run_inference_rag(author_name, user_prompt, **gen_cfg):
    results, annotation = search_prompt(user_prompt, author_name)
    results = rerank_chunks(user_prompt, results)
    final_prompt = build_generation_prompt(user_prompt, author_name, results, annotation)
    return run_inference(lora_pipe, final_prompt, **gen_cfg)

# --- STREAMLIT APP ---
if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.title("Settings")
selected_author_key = st.sidebar.selectbox("Select author:", list(AUTHORS.keys()))
selected_author_fullname = AUTHORS[selected_author_key]

use_rag = st.sidebar.checkbox("Use RAG")
compare_base_model = st.sidebar.checkbox("Compare with Base Model")

# Generation sliders
st.sidebar.subheader("Generation Config")
max_new_tokens = st.sidebar.slider("Max new tokens", 50, 1024, 300, step=50)
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.7, step=0.1)
top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
repetition_penalty = st.sidebar.slider("Repetition Penalty", 0.5, 2.0, 1.1, step=0.1)
top_k = st.sidebar.slider("Top-k", 0, 100, 50, step=5)

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

st.title("Mimic Literary Style LLM")
st.subheader(f"{selected_author_fullname} talking...")

user_prompt = st.text_input("Your message:", key="input_text")

gen_cfg = dict(
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
    repetition_penalty=repetition_penalty,
    top_k=top_k,
)

if st.button("Generate") and user_prompt.strip():
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.spinner("Generating..."):
        if use_rag:
            response = run_inference_rag(selected_author_key, user_prompt, **gen_cfg)
            st.session_state.messages.append({
                "role": "assistant",
                "author_key": selected_author_key,
                "content": response
            })
        else:
            if compare_base_model:
                response_lora = run_inference_lora(selected_author_key, user_prompt, **gen_cfg)
                response_base = run_inference_base(selected_author_key, user_prompt, **gen_cfg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "author_key": selected_author_key,
                    "content_lora": response_lora,
                    "content_base": response_base,
                    "compare": True
                })
            else:
                response = run_inference_lora(selected_author_key, user_prompt, **gen_cfg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "author_key": selected_author_key,
                    "content": response
                })

# --- RENDERING ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.write(f"You: {msg['content']}")
    else:
        author_fullname = AUTHORS.get(msg["author_key"], "Author")
        if msg.get("compare"):
            c1, c2 = st.columns(2)
            with c1:
                st.subheader(f"{author_fullname} (LoRA)")
                st.write(msg["content_lora"])
            with c2:
                st.subheader(f"{author_fullname} (Base)")
                st.write(msg["content_base"])
        else:
            st.subheader(author_fullname)
            st.write(msg["content"])
        st.markdown("---")
