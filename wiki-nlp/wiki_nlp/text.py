from dataclasses import dataclass
from typing import (
    Iterable,
    Iterator
)
import re

import spacy

from spacy.tokens.doc import Doc

PAT_ALPHABETIC = re.compile('(((?![\d])\w)+)', re.UNICODE)


@dataclass
class Token:
    token: str
    lemma: str
    pos: str


class TextProcessor:
    """
    TextProcessor is a class that can convert string data into 
    a stream of tokens.


    Methods
    -------
    tokenize(text: str) -> Iterator[Token]
        Tokenizes the text and returns an iterator of tokens.
        Only words are converted to tokens. Punctuation and digits are ignored.
        The POS tag and the lemma of the original word are also computed
        and stored inside each Token object. 
    """

    _REDUNDANT_PIPES = [
        "parser",
        "ner",
        "entity_linker",
        "entity_ruler",
        "textcat",
        "textcat_multilabel",
        "attribute_ruler",
        "sentecizer",
        "transformer"
    ]

    def __init__(self):
        self._pipeline = spacy.load(
            "de_core_news_md", exclude=self._REDUNDANT_PIPES)

    def tokenize(self, text: str) -> Iterator[Token]:
        """
        Parameters
        ----------
        text : str
            The string to tokenize
        """

        doc = self._pipeline(text)
        for token in doc:
            m = PAT_ALPHABETIC.match(token.lemma_.lower())
            if m is not None:
                yield Token(token=token.text, lemma=m.group(), pos=token.pos)
