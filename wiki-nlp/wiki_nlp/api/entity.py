from dataclasses import dataclass


@dataclass
class Paragraph:
    id: str
    document: str
    text: str
    position: int
    dmm_index: int
