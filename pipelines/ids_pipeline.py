from zenml import pipeline

from steps.save_to_mongo_step import save_books_to_mongo
from steps.similarity_chunk_step import similarity_chunk_step
from steps.generate_instruction_dataset import generate_instruction_dataset


@pipeline(enable_cache=False)
def ids_pipeline():
    chunked_passages = similarity_chunk_step(threshold=0.5)
    save_books_to_mongo(chunked_passages, db_name="chunks")
    generate_instruction_dataset(chunked_passages)
