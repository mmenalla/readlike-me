import os
import time
import openai
from uuid import uuid4
from dotenv import load_dotenv
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

load_dotenv()

# --- CONFIG ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "vdb1"
HF_DATASET = "megi/author-style-rag-annotated"
SPLIT = "train"
BATCH_SIZE = 50
RETRY_DELAY = 5
MAX_RETRIES = 3

# --- FUNCTIONS ---

def get_embedding(text: str) -> list[float]:
    resp = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return resp.data[0].embedding


def setup_qdrant(collection_name: str, vector_dim: int = 1536) -> QdrantClient:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
        print(f"Created collection: {collection_name}")
    else:
        print(f"Collection already exists: {collection_name}")
    return client


def build_points(records: list[dict]) -> list[PointStruct]:
    points = []
    for rec in records:
        text = rec["paragraph"]
        embedding = get_embedding(text)
        points.append(PointStruct(
            id=str(uuid4()),
            vector=embedding,
            payload={
                "author": rec["author"],
                "topics": rec.get("topics"),
                "moods": rec.get("moods"),
                "style_tags": rec.get("style_tags"),
                "paragraph": text
            }
        ))
    return points


def safe_upsert(client: QdrantClient, collection_name: str, points: list[PointStruct]):
    for attempt in range(MAX_RETRIES):
        try:
            client.upsert(collection_name=collection_name, points=points)
            return True
        except Exception as e:
            print(f"Upsert attempt {attempt+1} failed: {e}. Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
    print("Failed to upsert batch after retries.")
    return False


# --- MAIN ---
def main():
    ds = load_dataset(HF_DATASET, split=SPLIT)
    client = setup_qdrant(COLLECTION_NAME)

    total = len(ds)
    print(f"Total records: {total}")

    batch_records = []
    for idx, row in enumerate(ds):
        batch_records.append({
            "author": row["author"],
            "paragraph": row["paragraph"],
            "topics": row.get("topics"),
            "moods": row.get("moods"),
            "style_tags": row.get("style_tags")
        })

        if len(batch_records) >= BATCH_SIZE or idx == total - 1:
            points = build_points(batch_records)
            success = safe_upsert(client, COLLECTION_NAME, points)
            if success:
                print(f"Upserted records {idx+1-len(batch_records)} to {idx}")
            batch_records = []


if __name__ == "__main__":
    main()