from typing import Iterator
from dataclasses import dataclass

import pymongo
from bson import ObjectId


@dataclass
class Document:
    doc_id: str
    position: int
    title: str
    text: str


class DocumentStore:
    """
    DocumentStore provides an interface 
    for querying documents from a data repository. 

    Methods
    -------
    read_docs(count: int, last_id=None) -> Iterator[Document]
        Reads count documents from the underlying store and returns 
        an iterator of documents. 
        If last_id is set to None, then the retrieved documents will be 
        the first count documents found in the data store. 
        If last_id is not None, then the retrieved documents will begin 
        starting from the id after last_id. 

    """

    def read_docs(self, count: int, last_id=None) -> Iterator[Document]:
        """
        Parameters
        ----------
        count : int
            The number of documents to read 
        last_id : str = None  
            The last id read. If last id is not None, then the documents returned
            will start from the id after last_id

        Raises
        ------ 
        NotImplementedError 
            Raised if the receiver does not implement this method  
        """
        raise NotImplementedError()


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
