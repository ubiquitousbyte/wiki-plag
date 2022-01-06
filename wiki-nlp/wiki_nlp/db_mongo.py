from typing import Iterator

import pymongo
from bson import ObjectId

from wiki_nlp.db import (
    Document,
    DocumentStore
)


class MongoDocumentStore(DocumentStore):

    def __init__(self, uri: str):
        self._client = pymongo.MongoClient(uri)
        self._db = self._client.wikiplag

    def read_docs(self, count: int, last_id=None) -> Iterator[Document]:
        query = {
            "paragraphs": {
                "$exists": True
            }
        }
        if last_id is not None:
            if isinstance(last_id, str):
                oid = ObjectId(last_id)
            elif isinstance(last_id, ObjectId):
                oid = last_id
            query["_id"] = {"$gt": oid}

        cursor = self._db.documents.find(query).limit(count)

        for document in cursor:
            doc_id = str(document['_id'])
            for p in document['paragraphs']:
                text = p['text']
                yield Document(doc_id=doc_id, position=p['position'],
                               title=p['title'], text=text)

    def close(self):
        self._client.close()
