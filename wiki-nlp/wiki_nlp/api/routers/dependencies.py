
from wiki_nlp.api.services import plag_service
from wiki_nlp.api.db import (
    ParagraphStore,
    MongoParagraphStore
)


def get_paragraph_store() -> ParagraphStore:
    return MongoParagraphStore("mongodb://wikiplag:wikiplag2021@localhost:27017/wikiplag")


def get_plag_service() -> plag_service.IPlagService:
    return plag_service.PlagService(get_paragraph_store())
