from zenml import pipeline

from steps.dummy import dummy_start, dummy_end
from steps.emmbedding_step import upload_embeddings_to_qdrant
from steps.save_to_mongo_step import save_books_to_mongo
from steps.similarity_chunk_step import similarity_chunk_step


@pipeline(enable_cache=False)
def rag_pipeline():
    chunked_passages = similarity_chunk_step(threshold=0.4)
    save_books_to_mongo(chunked_passages, db_name="rag_books")
    upload_embeddings_to_qdrant(
        chunked_passages,
        collection_name="rag_books_collection"
    )
