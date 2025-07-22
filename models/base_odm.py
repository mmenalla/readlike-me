from bson import ObjectId
from db.mongo_manager import get_collection

class BaseODM:
    collection_name = None
    db_name = "goodreads"

    def __init__(self, **kwargs):
        self._id = kwargs.get("_id", ObjectId())

    def to_dict(self):
        raise NotImplementedError

    @classmethod
    def collection(cls):
        return get_collection(cls.db_name, cls.collection_name)

    def save(self):
        doc = self.to_dict()
        doc["_id"] = self._id
        self.collection().replace_one({"_id": self._id}, doc, upsert=True)

    @classmethod
    def find(cls, filter={}):
        return [cls(**doc) for doc in cls.collection().find(filter)]

    @classmethod
    def find_one(cls, filter={}):
        doc = cls.collection().find_one(filter)
        return cls(**doc) if doc else None
