import os
import openai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from uuid import uuid4

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

def get_embedding(text: str) -> list[float]:
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def setup_qdrant(collection_name: str, vector_dim: int) -> QdrantClient:
    client = QdrantClient(host=os.getenv("QDRANT_HOST"), port=os.getenv("QDRANT_PORT"))
    print(client.get_collections().collections)
    if collection_name not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
    return client


def build_points(docs: list[dict]) -> list[PointStruct]:
    points = []
    for doc in docs:
        text = f"{doc.get('title')} by {doc.get('author')}\n\n{doc.get('chunks', [])}"
        embedding = get_embedding(text)
        print(f"Embedding length: {len(embedding)}, first 5 values: {embedding[:5]}")
        points.append(PointStruct(
            id=str(uuid4()),
            vector=embedding,
            payload={
                "title": doc.get("title"),
                "author": doc.get("author"),
                "chunks": doc.get("chunks", []),
            }
        ))
    return points
