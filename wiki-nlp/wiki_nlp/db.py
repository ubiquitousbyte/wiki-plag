from typing import Iterator
from dataclasses import dataclass


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
