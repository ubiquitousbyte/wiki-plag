from typing import Tuple
from dataclasses import dataclass


@dataclass
class SparseDocument:
    id: str
    title: str


@dataclass
class Paragraph:
    id: str
    document: SparseDocument
    title: str
    text: str
    position: int
    index: int
    coordinates: Tuple[int, int]
