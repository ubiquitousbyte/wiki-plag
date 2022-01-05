from dataclasses import dataclass
from typing import List


@dataclass
class Paragraph:
    """A paragraph represents a portion of text found in a document"""
    title: str
    position: int
    text: str


@dataclass
class Document:
    """A text document consisting of multiple paragraphs"""
    id: str
    title: str
    source: str
    categories: List[str]
    paragraphs: List[Paragraph]
