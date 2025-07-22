from datetime import datetime

from zenml import pipeline
from steps.crawl_books_step import crawl_books
from steps.enrich_book_step import enrich_books
from steps.save_to_mongo import save_books_to_mongo


@pipeline
def goodreads_pipeline(user_id: str):
    books = crawl_books(user_id)
    enriched_books = enrich_books(books)
    save_books_to_mongo(books=enriched_books)
