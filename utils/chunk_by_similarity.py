from sentence_transformers import SentenceTransformer, util
import torch

def chunk_by_similarity(sentences, threshold=0.4):
    if not sentences:
        return []

    if len(sentences) == 1:
        return sentences  # Only one sentence, return as a single chunk

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, convert_to_tensor=True)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        e_prev = embeddings[i - 1].unsqueeze(0) if embeddings[i - 1].ndim == 1 else embeddings[i - 1:i]
        e_curr = embeddings[i].unsqueeze(0) if embeddings[i].ndim == 1 else embeddings[i:i+1]

        sim = util.cos_sim(e_prev, e_curr).item()
        if sim >= threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
