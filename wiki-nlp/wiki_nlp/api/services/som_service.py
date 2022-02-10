from typing import Iterable
import json

import torch

from wiki_nlp.api.entity import Paragraph
from wiki_nlp.api.db import ParagraphStore

from wiki_nlp.som.model import SOM


class ISOMService:

    def most_similar(self, vector: torch.FloatTensor) -> Iterable[Paragraph]:
        pass


class SOMService:

    def __init__(self, som_path: str, som_map_path: str,
                 store: ParagraphStore):
        self.model = SOM.from_file(som_path)
        with open(som_map_path, 'r') as fobj:
            self.activation_map = json.load(fobj)
        self.store = store

    def most_similar(self, vector: torch.FloatTensor) -> Iterable[Paragraph]:
        winner = self.model.winner(vector)
        candidates = self.activation_map[winner]
        return self.store.read_by_dmm_indices(candidates)
