from zenml import step
from typing import List, Dict
from utils.goodreads_utils import get_book_and_author_description, add_to_metadata
import time

@step
def enrich_books(books: List[Dict]) -> List[Dict]:
    enriched = []

    for book in books:
        if book["url"]:
            desc, bio = get_book_and_author_description(book["url"])
            time.sleep(1)  # polite scraping
            book["book_description"] = desc
            book["author_description"] = bio
        enriched.append(book)

    return enriched
