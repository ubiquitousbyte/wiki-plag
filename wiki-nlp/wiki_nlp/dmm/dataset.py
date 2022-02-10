from typing import Iterator, Iterable
from collections import Counter, OrderedDict

from torchtext.vocab import vocab as create_vocab
from torch.utils.data import Dataset as TorchDataset

import spacy

from wiki_nlp.dmm.samplers import NoiseSampler, WordSampler


SPACY_DE_PIPE = spacy.load("de_core_news_sm", exclude=[
    "parser",
    "ner",
    "entity_linker",
    "entity_ruler",
    "textcat",
    "textcat_multilabel",
    "attribute_ruler",
    "sentecizer",
    "transformer"
])


def lemmatize(texts: Iterable[str], n_workers: int) -> Iterator[Iterable[str]]:
    for doc in SPACY_DE_PIPE.pipe(texts, n_process=n_workers):
        yield [tok.lemma_.lower() for tok in doc
               if len(tok.lemma_.replace(" ", "")) != 0 and not tok.is_punct]


class Vocab:

    def __init__(self, ordered_dict: OrderedDict, min_freq: int,
                 noise_size: int, word_sampling_rate: float):
        self.vocab = create_vocab(ordered_dict, min_freq=min_freq)
        self.vocab.set_default_index(len(self.vocab))
        self.noise_size = noise_size
        self.noise_sampler = NoiseSampler(noise_size)
        self.noise_sampler.compute_dist(ordered_dict, self.vocab)

        self.word_sampler = WordSampler(sampling_rate=word_sampling_rate)
        self.word_sampler.compute_dist(ordered_dict)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, token: str) -> int:
        return self.vocab[token]

    def __contains__(self, token: str) -> bool:
        return token in self.vocab

    def itos(self, index: int) -> str:
        return self.vocab.lookup_token(index)


class Dataset(TorchDataset):

    def __init__(self, documents: Iterator[Iterable[str]], min_freq: int = 3,
                 noise_size: int = 8, word_sampling_rate: float = 0.001,
                 vocab: Vocab = None):

        self.dataset = []
        if not vocab:
            counter = Counter()
            for doc in documents:
                counter.update(doc)
                self.dataset.append(doc)

            sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
            sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)

            ordered_dict = OrderedDict(sorted_by_freq_tuples)
            self.vocab = Vocab(ordered_dict, min_freq,
                               noise_size, word_sampling_rate)
        else:
            for doc in documents:
                self.dataset.append(doc)
            self.vocab = vocab

        super(Dataset, self).__init__()

    def __getitem__(self, index):
        return self.dataset[index]

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)
