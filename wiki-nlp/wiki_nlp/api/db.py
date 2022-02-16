from typing import Iterable, Tuple

from wiki_nlp.api.entity import Paragraph, SparseDocument

import pymongo


class ParagraphStore:

    def read_by_dmm_indices(self, indices: Iterable[int]) -> Iterable[Paragraph]:
        pass

    def read_by_som_coordinates(self, coordinate: Tuple[int, int]) -> Iterable[Paragraph]:
        pass


class MongoParagraphStore:

    def __init__(self, uri: str):
        self.client = pymongo.MongoClient(uri)

    def _dict2paragraph(self, p: dict) -> Paragraph:
        return Paragraph(
            id=str(p['_id']),
            document=SparseDocument(
                id=str(p['document']['id']),
                title=p['document']['title']
            ),
            title=p['title'],
            text=p['text'],
            position=p['position'],
            index=p['index'],
            coordinates=p['coordinates']
        )

    def read_by_dmm_indices(self, indices: Iterable[int]) -> Iterable[Paragraph]:
        coll = self.client.wikiplag.nlp
        for p in coll.find({"index": {"$in": indices}}):
            yield self._dict2paragraph(p)

    def read_by_som_coordinates(self, coordinate: Tuple[int, int]) -> Iterable[Paragraph]:
        coll = self.client.wikiplag.nlp
        for p in coll.find({"coordinates": {"$all": coordinate}}):
            yield self._dict2paragraph(p)
