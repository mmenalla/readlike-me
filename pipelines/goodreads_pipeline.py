from zenml import pipeline
from steps.crawl_books_step import crawl_books
from steps.enrich_book_step import enrich_books
from steps.get_passages_step import get_passages
from steps.save_to_mongo_step import save_books_to_mongo


@pipeline(enable_cache=False)
def goodreads_pipeline(user_id: str):
    books = crawl_books(user_id)
    enriched_books_with_description = enrich_books(books)
    enriched_books_with_passages = get_passages(enriched_books_with_description,1000,10)
    save_books_to_mongo(books=enriched_books_with_passages, db_name="passages")
