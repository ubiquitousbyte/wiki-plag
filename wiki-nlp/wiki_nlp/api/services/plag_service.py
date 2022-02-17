from typing import Iterable
import pickle

from torch.nn.functional import cosine_similarity
import torch

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
        # Lemmatize document
        document = lemmatize([document], n_workers=0)
        # Infer a vector
        vector = self.dmpv.infer_vector(next(document))
        # Map the vector to a neuron in the self-organizing map
        winner = self.som.winner(vector)
        winner = int(winner[0]), int(winner[1])
        # Query all documents that were mapped to the same neuron during training
        candidates = list(self.store.read_by_som_coordinates(winner))
        # Extract their vectors
        vectors = self.dmpv.dv[list(map(lambda x: x.index, candidates))]
        # Compute cosine similarity
        sims = cosine_similarity(torch.from_numpy(
            vector), torch.from_numpy(vectors))
        # Get top most similar documents in the same cluster
        top = torch.topk(sims, k=n)
        for value, index in zip(top.values, top.indices):
            yield PlagCandidate(paragraph=candidates[index], similarity=value)
