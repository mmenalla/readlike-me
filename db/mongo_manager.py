from typing import Optional
from pymongo import MongoClient


class GoodreadsMongoClient:
    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.books_collection = self.db[db_name]

    def insert_books(self, books: list[dict]):
        if not books:
            return
        self.books_collection.insert_many(books)

    def close(self):
        self.client.close()
