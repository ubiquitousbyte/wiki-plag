from typing import Iterable

from wiki_nlp.api.entity import Paragraph

import pymongo


class ParagraphStore:

    def read_by_dmm_indices(self, indices: Iterable[int]) -> Iterable[Paragraph]:
        pass


class MongoParagraphStore:

    def __init__(self, uri: str):
        self.client = pymongo.MongoClient(uri)

    def read_by_dmm_indices(self, indices: Iterable[int]) -> Iterable[Paragraph]:
        coll = self.client.wikiplag.nlp
        for p in coll.find({"dmmIndex": {"$in": indices}}):
            yield Paragraph(id=str(p['_id']), document=str(p['document']), text=p['text'],
                            position=p['position'], dmm_index=p['dmmIndex'])
