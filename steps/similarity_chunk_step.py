import os

from zenml import step
from typing import List, Dict

from db.mongo_manager import GoodreadsMongoClient
from utils.chunk_by_similarity import chunk_by_similarity

@step
def similarity_chunk_step(
    threshold: float = 0.4
) -> list[dict]:
    mongo_uri = os.environ.get("MONGO_URI")
    client = GoodreadsMongoClient(uri=mongo_uri, db_name="passages")
    books = list(client.books_collection.find())

    chunked_passages = []
    for book in books:
        if "passages" not in book or not book["passages"]:
            print(f"Skipping book with no passages: {book.get('title', 'Unknown')}")
            continue
        sentences = book["passages"]
        chunks = chunk_by_similarity(sentences, threshold=threshold)
        rec = {
            "author": book["author"],
            "title": book["title"],
            "chunks": chunks
        }
        chunked_passages.append(rec)

    return chunked_passages