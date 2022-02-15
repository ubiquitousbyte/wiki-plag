from typing import Tuple

from pydantic import BaseModel, validator

from wiki_nlp.api.entity import Paragraph


class PlagCandidate(BaseModel):
    paragraph: Paragraph
    similarity: float
    coordinate: Tuple[int, int]


class Text(BaseModel):
    text: str

    @validator('text')
    def text_must_have_atleast_64_characters(cls, v):
        if len(v) < 64:
            raise ValueError("Text must contain at least 64 characters")
        return v
