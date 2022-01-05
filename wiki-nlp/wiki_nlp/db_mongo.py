from typing import AsyncIterator
import asyncio

import motor.motor_asyncio as mongo

from . import (
    db,
    entity
)


class MongoDocumentStore(db.DocumentStore):

    def __init__(self, uri: str):
        self._client = mongo.AsyncIOMotorClient(uri)
        self._db = self._client.wikiplag

    async def read_docs(self, count: int, last_id=None):
        query = {
            "paragraphs": {
                "$exists": True
            }
        }
        if last_id is not None:
            query["_id"] = {"$gt": last_id}

        cursor = self._db.documents.find(query).limit(count)
        async for document in cursor:
            print(document)

    def close(self):
        self._client.close()


if __name__ == '__main__':
    store = MongoDocumentStore(
        "mongodb://wikiplag:wikiplag2021@localhost:27017/wikiplag")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(store.read_docs(10))
