from collections import Counter

from torchtext.vocab import Vocab

import numpy as np
from numpy.random import default_rng


class NoiseSampler:

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
