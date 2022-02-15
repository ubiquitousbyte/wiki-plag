from dataclasses import dataclass


@dataclass
class SparseDocument:
    id: str
    title: str

    def __post_init__(self):
        if not isinstance(self.id, str):
            self.id = str(self.id)


@dataclass
class Paragraph:
    id: str
    document: SparseDocument
    title: str
    text: str
    position: int
    index: int

    def __post__init__(self):
        if not isinstance(self.id, str):
            self._id = str(self.id)
