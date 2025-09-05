from zenml import step
from typing import List, Dict
from db.mongo_manager import GoodreadsMongoClient
import os


@step
def save_books_to_mongo(books: List[Dict], db_name) -> None:
    mongo_uri = os.getenv("MONGO_URI")
    client = GoodreadsMongoClient(uri=mongo_uri, db_name=db_name)
    client.insert_books(books)
    client.close()
