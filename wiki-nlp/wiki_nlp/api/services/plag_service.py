from typing import Iterable
import pickle
import numpy as np

from wiki_nlp.api.services.schema import PlagCandidate
from wiki_nlp.api.db import ParagraphStore
from wiki_nlp.dmm.dataset import lemmatize

from gensim.models.doc2vec import Doc2Vec as DMM
from minisom import MiniSom


class IPlagService:

    def find_candidates(self, document: str, n: int = 4) -> Iterable[PlagCandidate]:
        pass


class PlagService:

    def __init__(self, store: ParagraphStore):
        self.store = store
        self.dmpv: DMM = DMM.load("dmm_2.pth")
        with open("som_2.pth", 'rb') as sfile:
            self.som: MiniSom = pickle.load(sfile)

    def find_candidates(self, document: str, n: int = 4) -> Iterable[PlagCandidate]:
        # Lemmatise document
        document = lemmatize([document], n_workers=1)

        # Infer a vector
        vector = self.dmpv.infer_vector(next(document))

        # Find documents in the dataset most similar to the inferred vector
        sims = self.dmpv.dv.most_similar([vector], topn=n)

        # Extract those documents from the database
        ps = self.store.read_by_dmm_indices(list(map(lambda x: x[0], sims)))

        # Sort data by index
        sims = sorted(sims, key=lambda pair: pair[0])
        ps = sorted(ps, key=lambda p: p.index)

        # Group documents and similarities
        for (_, similarity), p in zip(sims, ps):
            p.id = str(p.id)
            yield PlagCandidate(paragraph=p, similarity=similarity)
