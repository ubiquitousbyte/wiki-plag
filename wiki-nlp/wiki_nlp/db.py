from typing import AsyncIterator

from wiki_nlp.entity import Document


class DocumentStore:
    """
    DocumentStore provides an interface 
    for asynchronously querying document batches from a data repository. 
    """
