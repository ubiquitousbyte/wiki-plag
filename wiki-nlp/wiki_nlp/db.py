from typing import AsyncIterator

from wiki_nlp.entity import Document


class DocumentStore:
    """
    DocumentStore provides an interface 
    for asynchronously querying documents from a data repository. 

    Methods
    -------
    read_docs(count: int, last_id=None) -> AsyncIterator[Document]
        Reads count documents from the underlying store and returns 
        an asynchronous iterator of documents. 
        If last_id is set to None, then the retrieved documents will be 
        the first count documents found in the data store. 
        If last_id is not None, then the retrieved documents will begin 
        starting from the id after last_id. 

    """

    async def read_docs(self, count: int, last_id=None) -> AsyncIterator[Document]:
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
