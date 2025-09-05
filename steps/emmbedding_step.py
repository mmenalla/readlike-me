from zenml import step
from typing import List, Dict
from utils.embedding_utils import setup_qdrant, get_embedding, build_points


@step
def upload_embeddings_to_qdrant(docs: List[Dict], collection_name: str) -> None:
    vector_dim = len(get_embedding("test"))
    client = setup_qdrant(collection_name, vector_dim)
    points = build_points(docs)
    print(f"Uploading {len(points)} points to Qdrant")
    result = client.upsert(collection_name=collection_name, points=points)
    print(result)
