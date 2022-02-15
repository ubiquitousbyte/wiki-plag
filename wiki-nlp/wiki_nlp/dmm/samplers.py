from collections import Counter

from torchtext.vocab import Vocab

import numpy as np
from numpy.random import default_rng


class NoiseSampler:
    """
    The noise sampler randomly samples words from a vocabulary using a
    power-law distribution that favours infrequent words.
    The sampler allows us to avoid computing the softmax normalizing constant
    which is an O(n) operation.
    With the noise sampler we can replace softmax with a non-linear logicstic
    regression that discriminates the observed data from artificially
    generated noise.

    Methods
    -------
    sample()
        Returns a list of noise samples.
        The noise samples are represented as indices to words in the
        vocabulary.
    """

    def __init__(self, size: int):
        self._noise_size = size
        self._rng = default_rng()

    def compute_dist(self, frequencies: Counter, vocab: Vocab):
        self._dist = np.zeros((len(frequencies)+1, ))
        for word, freq in frequencies.items():
            self._dist[vocab[word]] = freq
        self._dist = np.power(self._dist, 0.75)
        self._dist /= np.sum(self._dist)

    def sample(self):
        return self._rng.choice(self._dist.shape[0],
                                self._noise_size, p=self._dist).tolist()


class WordSampler:
    """
    A word sampler that assigns probabilities to every word in the vocabulary.
    Overfrequent words are assigned low probabilities to prevent
    training a model that pushes unique words towards ordinary ones.
    Conversely, infrequent words are assigned high sampling probabilities
    to ensure that they obtain meaningful representations in the embedding space.

    Attributes
    ----------
    sampling_rate: float
        The lower the sampling rate, the stricter the sampler enforces
        uniqueness.

    Methods
    -------
    compute_dist(frequencies: Counter)
        Computes the sampling distribution from the frequency distribution
        of all words in the vocabulary 

    __contains__(word: str) -> bool
        Returns true if the sampler determines that the word is unique
        enough to be used during training.
    """

    def __init__(self, sampling_rate: float = 0.001):
        self.sampling_rate = sampling_rate

    def compute_dist(self, frequencies: Counter):
        self._sampler = {}
        wc = len(frequencies)
        for word, freq in frequencies.items():
            f = freq/wc
            p = (np.sqrt(f/self.sampling_rate)+1)*(self.sampling_rate/f)
            self._sampler[word] = p

    def __contains__(self, word: str) -> bool:
        return self._sampler[word] > np.random.random_sample()
