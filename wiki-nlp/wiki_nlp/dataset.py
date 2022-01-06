from typing import (
    Iterable,
    OrderedDict,
    Tuple,
    List,
    Counter
)


from torch.utils.data import dataset
from torchtext.vocab import vocab

from wiki_nlp import db
from wiki_nlp.text import (
    Token,
    TextProcessor
)


class Vocabulary:
    """
    Vocabulary is a class that containts a known set of tokens.

    Methods
    -------
    build(tokens: Iterable[Token])
        Constructs the vocabulary from an iterable of tokens. 
        If use_lemmas was set, the lemmatized version of each token 
        will be added to the vocabulary.
        Otherwise, the standard version will be added

    __getitem__(token: str) -> int
        Returns the index in the vocabulary associated with the specified token

    __contains__(token: str) -> bool
        Returns true if the vocabulary knows the token 

    __len__() -> int
        Returns the number of tokens in the vocabulary
    """

    def __init__(self):
        self.counter = Counter()

    def build(self, tokens: Iterable[Token], use_lemmas: bool, min_freq: int):
        """
        Parameters
        ----------
        tokens : Iterable[Token]
            The tokens to construct the vocabulary from
        use_lemmas : bool 
            If True, the vocabulary will use the lematized versions of 
            the tokens to construct itself 
        min_freq : int 
            The minimal amount of times a token needs to occur 
            to be added by the vocabulary. 
        """

        if use_lemmas:
            self.counter.update([t.lemma for t in tokens])
        else:
            self.counter.update([t.token for t in tokens])
        self.vocab = vocab(OrderedDict(
            self.counter.items()), min_freq=min_freq)
        self.vocab.set_default_index(len(self.vocab))

    def __len__(self) -> int:
        return len(self.vocab)

    def __contains__(self, token: str) -> bool:
        return token in self.vocab

    def __getitem__(self, token: str) -> int:
        return self.vocab[token]

    def get_default_index(self) -> int:
        return len(self.vocab)


class Dataset(dataset.Dataset):
    """
    Dataset represents an in-memory dataset of documents.
    Each document is represented as a list of tokens

    Methods
    -------
    __getitem__(index : int) -> List[Token]
        Returns the document at the given index 

    __len__() -> int 
        Returns the length of the dataset 

    get_document(index : int) -> Tuple[str, str]
        Returns a tuple containing the document identifier and the document title
        associated with the sample in the dataset at the given index 
    """

    def __init__(self, store: db.DocumentStore, size: int):
        """
        Parameters
        ----------
        store : DocumentStore
            The document store to construct the dataset from 
        size : int 
            The desired size of the dataset. This represents the number 
            of documents that will be extracted from the document store.
        """

        super(Dataset, self).__init__()
        processor = TextProcessor()
        self._dataset = []
        self._paragraphs = {}
        for sample in store.read_docs(size):
            sample_tokens = list(processor.tokenize(sample.text))
            if len(sample_tokens) > 0:
                self._dataset.append(sample_tokens)
                self._paragraphs[len(self._dataset) -
                                 1] = (sample.doc_id, sample.title)

    def __getitem__(self, index: int) -> List[Token]:
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)

    def get_document(self, index: int) -> Tuple[str, str]:
        return self._paragraphs[index]
