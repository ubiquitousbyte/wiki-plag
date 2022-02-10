from typing import List, Iterable

import torch

from wiki_nlp.api.entity import Paragraph
from wiki_nlp.api.db import MongoParagraphStore, ParagraphStore

from wiki_nlp.dmm.model import DMM
from wiki_nlp.dmm.dataset import lemmatize, Dataset


class IDMMService:

    def predict_vector(self, text: str) -> torch.FloatTensor:
        pass

    def most_similar(self, text: str, topn: int = 4) -> Iterable[Paragraph]:
        pass


class DMMService(IDMMService):

    def __init__(self, store: ParagraphStore, dmm_path: str, dmm_vocab_path: str):
        self.model = DMM.from_file(dmm_path)
        self.vocab = torch.load(dmm_vocab_path)
        self.store = store

    def predict_vector(self, text: str) -> torch.FloatTensor:
        data = list(lemmatize([text], n_workers=1))
        test_set = Dataset(documents=data, vocab=self.vocab)
        return self.model.predict(dataset=test_set, epochs=20, n_workers=1, ctx_size=2)

    def most_similar(self, text: str, topn: int = 9) -> Iterable[Paragraph]:
        vector = self.predict_vector(text)
        dists, indices = self.model.most_similar(vector, topn=topn)
        print(dists)
        paragraphs = list(self.store.read_by_dmm_indices(indices.tolist()))
        return paragraphs
