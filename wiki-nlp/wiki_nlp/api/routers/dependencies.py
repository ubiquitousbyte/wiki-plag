
from wiki_nlp.api.services import dmm_service, som_service
from wiki_nlp.api.db import (
    ParagraphStore,
    MongoParagraphStore
)


def get_dmm_path() -> str:
    return "dmm_1.pth"


def get_dmm_vocab_path() -> str:
    return "dmm_vocab_1.pth"


def get_som_path() -> str:
    return "som_1.pth"


def get_som_map_path() -> str:
    return "som_map_1.json"


def get_paragraph_store() -> ParagraphStore:
    return MongoParagraphStore("mongodb://wikiplag:wikiplag2021@localhost:27017/wikiplag")


def get_dmm_service() -> dmm_service.IDMMService:
    store = get_paragraph_store()
    dmm_path = get_dmm_path()
    vocab_path = get_dmm_vocab_path()
    return dmm_service.DMMService(store, dmm_path, vocab_path)


def get_som_service() -> som_service.ISOMService:
    store = get_paragraph_store()
    som_path = get_som_path()
    map_path = get_som_map_path()
    return som_service.SOMService(som_path, map_path, store)
