
from wiki_nlp.api.services import plag_service
from wiki_nlp.api.db import (
    ParagraphStore,
    MongoParagraphStore
)


def get_db_pwd() -> str:
    with open("/run/secrets/db-password", "r") as fobj:
        return fobj.read()


def get_paragraph_store() -> ParagraphStore:
    password = get_db_pwd()
    return MongoParagraphStore(f"mongodb://wikiplag:{password}@localhost:27017/wikiplag")


def get_plag_service() -> plag_service.IPlagService:
    return plag_service.PlagService(get_paragraph_store())
