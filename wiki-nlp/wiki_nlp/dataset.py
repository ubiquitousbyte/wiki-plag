from typing import (
    Iterable,
    Iterator,
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

    def build(self, tokens: Iterable[Token], min_freq: int):
        """
        Parameters
        ----------
        tokens : Iterable[Token]
            The tokens to construct the vocabulary from
        min_freq : int 
            The minimal amount of times a token needs to occur 
            to be added by the vocabulary. 
        """

        self.counter.update([t.data for t in tokens])
        self.vocab = vocab(OrderedDict(self.counter.items()), min_freq)
        self.vocab.set_default_index(len(self.vocab))

    def __len__(self) -> int:
        return len(self.vocab)

    def __contains__(self, token: str) -> bool:
        return token in self.vocab

    def __getitem__(self, token: str) -> int:
        return self.vocab[token]

    def get_default_index(self) -> int:
        return len(self.vocab)

    def itos(self, index: int) -> str:
        return self.vocab.lookup_token(index)


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

    def __init__(self, data: Iterable[Iterable[Token]]):
        """
        Parameters
        ----------
        data : Iterable[Iterable[Token]]
            An iterable of documents.
            Each document is represented as an iterable of tokens
        """

        super(Dataset, self).__init__()
        self._dataset = data

    @classmethod
    def from_documents(cls, documents: Iterator[db.Document],
                       use_lemmas: bool) -> 'Dataset':
        """
        Constructs a dataset from an iterator of database documents

        Parameters
        ----------
        documents : Iterator[db.Document]
            The documents to construct the dataset from 

        use_lemmas : bool
            Set to true if the dataset should be constructed from the lemmatized
            versions of each token found in a document
        """

        processor = TextProcessor()
        data = []
        for example in documents:
            tokens = list(processor.tokenize(example.text, use_lemmas))
            if len(tokens) > 0:
                data.append(tokens)
        return cls(data)

    def __getitem__(self, index: int) -> List[Token]:
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)
