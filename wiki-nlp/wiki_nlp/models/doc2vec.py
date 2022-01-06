from typing import (
    Optional,
    Callable,
    Tuple,
    List
)
import multiprocessing
import os
import signal
import math

import torch
import torch.nn as nn
import numpy as np
from numpy.random import default_rng

from wiki_nlp.dataset import (
    Vocabulary,
    Dataset
)

from wiki_nlp.text import Token


class DMM(nn.Module):
    """
    A class used to represent an implementation of the 
    Distributed Memory Model of Paragraph Vectors proposed by
    Mikholov et. al. in Distributed Representations of Sentences and Documents.

    The original paper can be found at https://arxiv.org/abs/1405.4053

    Attributes
    ----------
    document_count : int
        The number of documents used for training the model. 
        This parameter is used to construct the document matrix. 
    vocabulary_size : int 
        The size of the vocabulary. This parameter is used to construct 
        the input word matrix. 
    embedding_size : int
        The size of the embedding space.

    Methods
    -------
    forward(ctxs: torch.FloatTensor, docs: torch.FloatTensor, y: torch.FloatTensor)
        Computes the forward step of the model
    """

    def __init__(self, document_count: int, vocabulary_size: int, embedding_size: int):
        """
        Parameters
        ----------
        document_count : int
            The number of documents used for training the model. 
            This parameter is used to construct the document matrix. 
        vocabulary_size : int 
            The size of the vocabulary. This parameter is used to construct 
            the input word matrix. 
        embedding_size : int
            The size of the embedding space.
        """

        super(DMM, self).__init__()
        self._D = nn.Parameter(
            torch.randn(document_count, embedding_size), requires_grad=True
        )
        self._W = nn.Parameter(
            torch.cat((torch.randn(vocabulary_size, embedding_size),
                      torch.zeros(1, embedding_size))),
            requires_grad=True
        )
        self._WP = nn.Parameter(
            torch.randn(embedding_size, vocabulary_size), requires_grad=True
        )

    def forward(self, ctxs: torch.FloatTensor, docs: torch.FloatTensor,
                y: torch.FloatTensor):
        """
        Parameters
        ----------
        ctxs : torch.FloatTensor
            A matrix that maps context word vectors to the documents
            in which the context words occur. 
        docs : torch.FloatTensor 
            A matrix representing a batch of documents that will be linearly
            combined with their respective context word vectors
        y : torch.FloatTensor
            A matrix consisting of vectors that hold noise words and the
            target word (the current center word) for each document.
            The target word must be the first element in the matrix. 

        Example
        -------
        Let docs be a batch consisting of 5 documents.
        Let ctxs be a word vector batch of 5 contexts (one for each document), 
        where each context consists of 10 context words.
        Let the number of noise samples per document be 10. 

        The forward pass consists of two phases.

        Phase 1 (Input -> Hidden)
        -------------------------
        We perform a linear transformation on the context vectors and the 
        document vectors. 
        Assuming an embedding size of 100, we compute:

        [5 x 100] + [5 x ∑ [10 x 100]] = [5 x 100] + [5 x 100] = [5 x 100]

        The output of this computation represents the hidden state h for each
        of the 5 documents in the batch.

        Phase 2 (Hidden -> Output)
        --------------------------
        We compute the similarity between the hidden state and the target word,
        as well as between the hidden state and the noise samples. 
        Assuming an embedding size of 100, we compute:

        [5 x 1 x 100] x [100 x 5 x 11] = [5 x 1 x 11]

        We remove the redundant dimension and [5 x 1 x 11] becomes [5 x 11]. 
        Thus, for each of the documents in the batch, we obtain a similarity
        metric to its target word (1) and its context words (10).
        """

        h = torch.add(self._D[docs, :], torch.sum(self._W[ctxs, :], dim=1))
        return torch.bmm(h.unsqueeze(1), self._WP[:, y].permute(1, 0, 2)).squeeze()


class Loss(nn.Module):
    """
    The negative sampling loss function used to compute the 
    distance between the document vector to its target word 
    relative to its distance from a set of noise samples 

    Methods
    -------
    forward(scores: torch.FloatTensor)
        Computes the negative sampling loss of the scores.
    """

    def __init__(self):
        super(Loss, self).__init__()
        self._loss = nn.LogSigmoid()

    def forward(self, scores: torch.FloatTensor):
        """
        Parameters
        ----------
        scores: torch.FloatTensor
            The score matrix mapping documents to their similarities to 
            their current target word and the noise samples.
            The first index in the scores matrix must be the similarity
            between the document vector and its target word.
            All other indices must represent the similarity between the document
            vector and the negative samples.
        """

        n = scores.size()[0]
        k = scores.size()[1] - 1
        return -torch.sum(
            self._loss(scores[:, 0]) +
            torch.sum(self._loss(-scores[:, 1:])) / k) / n


class WordSampler:
    """
    A word sampler that assigns probabilities to every word in the vocabulary.
    Overfrequent words are assigned low probabilities to prevent
    training a model that pushes unique words towards ordinary ones. 
    Conversely, infrequent words are assigned high sampling probabilities
    to ensure that they obtain meaningful representations in the embedding space.

    Attributes
    ----------
    vocabulary_size : int
        The number of words in the vocabulary
    sampling_rate : float 
        The higher the sampling rate, the stricter the sampler enforces 
        uniqueness. 

    Methods
    -------
    compute_dist(vocab: Vocabulary)
        Computes the sampling distribution for all words in the vocabulary

    __contains__(word: str) -> bool
        Returns true if the sampler determines that the word is unique 
        enough to be used during training. 
    """

    def __init__(self, sampling_rate: float = 0.001):
        self.sampling_rate = sampling_rate

    def compute_dist(self, vocab: Vocabulary):
        self._sampler = {}
        wc = len(vocab)
        for word, freq in vocab.counter.items():
            f = freq/wc
            p = (np.sqrt(f/self.sampling_rate)+1)*(self.sampling_rate/f)
            self._sampler[word] = p

    def __contains__(self, word: str) -> bool:
        return self._sampler[word] > np.random.random_sample()


class NoiseSampler:
    """
    The noise sampler randomly samples words from a vocabulary using a 
    power-law distribution that favours infrequent words.

    The sampler allows us to avoid computing the softmax normalizing constant
    which is an O(n) operation. 

    With the noise sampler we can replace softmax with a non-linear logicstic 
    regression that discriminates the observed data from artificially 
    generated noise.

    """

    def __init__(self, size: int):
        self._noise_size = size
        self._rng = default_rng()

    def compute_dist(self, vocab: Vocabulary):
        self._dist = np.zeros((len(vocab)),)
        for word, freq in vocab.counter.items():
            self._dist[vocab[word]] = freq
        self._dist = np.power(self._dist, 0.75)
        self._dist /= np.sum(self._dist)

    def sample(self):
        return self._rng.choice(self._dist.shape[0],
                                self._noise_size, p=self._dist).tolist()


class _Batch:
    """
    This class represents a batch of training examples.
    The batch consists of a set of documents, the noise samples to use 
    for the documents during training, as well as the context words to 
    aggregate with each document. 

    Attributes
    ----------
    ctxs : List[List[int]]
        A set of context words for each document in the batch 
        The outer list represents a mapping between docs and their context 
        words. 
        In other words, the invariant len(ctxs) = len(docs) always holds. 

    docs : List[int]
        A set of documents that are a part of the batch

    y : List[List[int]]
        The noise samples and the target word for each document in the batch
        The outer list represents a mapping between docs, the target word 
        for each doc, and the noise samples for each word.
        Similarly to ctxs, the invariant len(y) = len(docs) always holds. 

    Methods
    -------
    torchify()
        Convertes the attributes to tensors

    cudify()
        Converts the attributes to CUDA vectors
    """

    def __init__(self, ctx_size: int):
        self.ctxs = [] if ctx_size > 0 else None
        self.docs = []
        self.y = []

    def __len__(self):
        """Returns the size of the batch"""
        return len(self.docs)

    def torchify(self):
        """Converts the batch attributes to tensors"""
        self.ctxs = torch.LongTensor(self.ctxs)
        self.docs = torch.LongTensor(self.docs)
        self.y = torch.LongTensor(self.y)

    def cudify(self):
        """Converts the batch attributes to CUDA vectors"""
        if self.ctxs is not None:
            self.ctxs = self.ctxs.cuda()
        self.docs = self.docs.cuda()
        self.y = self.y.cuda()


"""
A function pointer that, given a document (a list of tokens), counts 
the number of words that are still to be processed relative to the 
index of the current center word. 

The current center word need not be specified. If it is not, then 
all the examples returned equal the length of the list minus total 
the context size.
"""
ExampleCounter = Callable[[List[Token], Optional[int]], int]


class _BatchState:
    """
    This class represents the state of a batch while it gets constructed 
    from a vocabulary and dataset. 

    The batch state consists of the current document being processed,
    the current word in the document being processed and a context size
    to offset words. 

    Methods
    -------
    forward(dataset: Dataset, batch_size: int, e: ExampleCounter) -> Tuple[int, int]
        Returns the current document and word being processed and advances
        the batch state to the next word/document. 

        The example counter is used to delimit words and documents. 
        The batch size defines the number of documents to put in a single batch. 
        The dataset holds the documents to use while constructing the batch
    """

    def __init__(self, ctx_size: int):
        self._doc_index = multiprocessing.RawValue('i', 0)
        self._word_index = multiprocessing.RawValue('i', ctx_size)
        self._mutex = multiprocessing.Lock()
        self._ctx_size = ctx_size

    def forward(self, dataset: Dataset, batch_size: int,
                e: ExampleCounter) -> Tuple[int, int]:
        """
        This function is thread-safe. 

        Parameters
        ----------
        dataset : Dataset
            The set of documents to construct the batch from
        batch_size : int 
            The number of documents to put in a single batch
        e : ExampleCounter
            Counts the number of examples in a document and uses it
            to delimit words and documents in the batch 
        """

        with self._mutex:
            doc_index, word_index = self._doc_index.value, self._word_index.value
            self._forward(dataset, batch_size, self._ctx_size, e)
            return doc_index, word_index

    def _forward(self, dataset: Dataset, batch_size: int, e: ExampleCounter):
        """
        This function is NOT thread-safe. 

        Parameters
        ----------
        dataset : Dataset
            The set of documents to construct the batch from
        batch_size : int 
            The number of documents to put in a single batch
        e : ExampleCounter
            Counts the number of examples in a document and uses it
            to delimit words and documents in the batch 
        """

        ex_count = e(dataset[self._doc_index.value], self._word_index.value)

        if ex_count > batch_size:
            # Successfully processed a word batch
            # Position the current word index to the start
            # of the next word batch in the current document and return
            self._word_index.value += batch_size
            return

        if ex_count == batch_size:
            # Successfully processed a document batch
            if self._doc_index.value < len(dataset) - 1:
                # If there are more documents in the dataset
                # advance the pointer to the next document
                self._doc_index.value += 1
            else:
                # If there aren't any more documents
                # position the pointer to the start of the dataset
                self._doc_index.value = 0
            # Reset the word pointer to the context size
            self._word_index.value = self._ctx_size
            return

        while ex_count < batch_size:
            # We haven't accumulated enough examples
            # to fit in the batch

            if self._doc_index.value == len(dataset) - 1:
                # If the current document pointer points
                # to the last document, reset the pointer
                # to the start of the dataset
                self._doc_index.value = 0
                self._word_index.value = self._ctx_size
                return

            # The current document had an insufficient number of words
            # to populate the batch.
            # Advance to the next document
            self._doc_index.value += 1
            # Accumulate examples from the next document
            ex_count += e(dataset[self._doc_index.value])

        # Set the word index to the start of the next batch
        self._word_index.value = (len(dataset[self._doc_index.value])
                                  - self._ctx_size
                                  - (ex_count - batch_size))


class _BatchGenerator:
    """
    A synchronous batch generator that yields batches of inputs to train 
    a Doc2Vec model.

    Attributes
    ----------
    vocab : Vocabulary 
        The vocabulary to sample words from. 
    dataset : Dataset 
        The dataset of documents to yield batches from
    batch_size : int 
        The number of samples to put in a single batch 
    ctx_size: int 
        The relative context size of a center word. 
        The size is relative because it only denotes the number of context 
        words on ONE side of a center word.
        Therefore, total context size is actually 2*ctx_size
    noise_size : int
        The number of noise words to create per sample in the batch

    Methods
    -------
    forward() -> _Batch:
        Populates a batch and returns it to the caller 

    example_counter(doc: List[Token], word_index=None)
        Counts the number of examples in the document relative to the word index

    _fill_batch(batch: _Batch, doc_index: int, word_index: int)
        Inserts a single element into the batch.
        The element is defined by the current document being processed 
        and the index of the current center word being processed.

    __len__() -> int:
        Returns the total number of batches that can be generated
    """

    def __init__(self, vocab: Vocabulary, dataset: Dataset,
                 batch_size: int, ctx_size: int, noise_size: int):

        self.dataset = dataset
        self.batch_size = batch_size
        self.ctx_size = ctx_size

        self.vocab = vocab

        self.noise_sampler = NoiseSampler(noise_size)
        self.noise_sampler.compute_dist(vocab)

        self.word_sampler = WordSampler()
        self.word_sampler.compute_dist(vocab)

        self._state = _BatchState(self.ctx_size)

    def forward(self) -> _Batch:
        doc_index, word_index = self._state.forward(
            self.dataset, self.batch_size, self.example_counter)

        batch = _Batch(self.ctx_size)

        while len(batch) < self.batch_size:
            if doc_index == len(self.dataset):
                # All documents have been processed
                # Torchify the batch and return it
                batch.torchify()
                break

            # Compute the remaining number of context words
            # in the current document
            remaining = len(self.dataset[doc_index]) - 1 - self.ctx_size
            if word_index <= remaining:
                # Extract the index of the current word in the vocabulary
                vi = self.vocab[self.dataset[doc_index][word_index]] - 1

                if self.vocab.itos(vi) in self.word_sampler:
                    # The word sampler has determined that the
                    # current word is unique enough to be included
                    # in the batch
                    self._fill_batch(batch, doc_index, word_index)

                # Advance to the next word
                word_index += 1
            else:
                # All context words for this document have been processed
                # We therefore advance to the next document
                doc_index += 1
                word_index = self.ctx_size

        # The batch has been filled
        # Torchify the batch and return it
        batch.torchify()
        return batch

    def example_counter(self, doc: List[Token], word_index=None):
        if word_index is not None:
            if len(doc) - word_index >= self.ctx_size + 1:
                return len(doc) - word_index - self.ctx_size
            return 0

        if len(doc) >= 2 * self.ctx_size+1:
            return len(doc) - 2 * self.ctx_size
        return 0

    def _fill_batch(self, batch: _Batch, doc_index: int, word_index: int):
        # Add the document to the batch
        batch.docs.append(doc_index)

        # Sample from the noise distribution
        noise = self.noise_sampler.sample()

        doc = self.dataset[doc_index]

        # Add the target word to the noise as the first index
        noise.insert(0, self.vocab[doc[word_index]] - 1)
        # Add the noise and target word to the batch
        batch.y.append(noise)

        if self.ctx_size == 0:
            return

        # Construct the context
        ctx = []
        ctx_indexes = (word_index + diff for diff in
                       range(-self.ctx_size, self.ctx_size+1)
                       if diff != 0)
        for i in ctx_indexes:
            index = self.vocab[doc[i]] - 1
            ctx.append(index)
        batch.ctxs.append(ctx)

    def __len__(self):
        examples = sum(self.example_counter(d) for d in self.dataset)
        return math.ceil(examples / self.batch_size)


class BatchGenerator:
    """
    A concurrent batch generator that constructs document batches to feed
    in a Doc2Vec model. 


    Attributes
    ----------
    vocab : Vocabulary 
        The vocabulary to sample words from. 
    dataset : Dataset 
        The dataset of documents to yield batches from
    batch_size : int 
        The number of samples to put in a single batch 
    ctx_size: int 
        The relative context size of a center word. 
        The size is relative because it only denotes the number of context 
        words on ONE side of a center word.
        Therefore, total context size is actually 2*ctx_size
    noise_size : int
        The number of noise words to create per sample in the batch
    workers : int
        The number of processes to use for batch generation 

    Methods
    -------
    start():
        Spawns all workers that create batches

    stop()
        Stops all workers 

    forward()
        Returns a batch when ready

    is_running() -> bool
        Returns true if the batch generator is running
    """

    def __init__(self, vocab: Vocabulary, dataset: Dataset, batch_size: int,
                 ctx_size: int, noise_size: int, max_size: int, workers: int):
        self.max_size = max_size
        self.num_workers = (workers if workers > 0
                            else multiprocessing.cpu_count())

        self._generator = _BatchGenerator(vocab, dataset, batch_size,
                                          ctx_size, noise_size)

        self._queue = None
        self._stop_event = None
        self._processes = []

    def start(self):
        self._queue = multiprocessing.Queue(maxsize=self.max_size)
        self._stop_event = multiprocessing.Event()

        for _ in range(self.num_workers):
            worker = multiprocessing.Process(target=self._work)
            worker.daemon = True
            self._processes.append(worker)
            worker.start()

    def _work(self):
        while not self._stop_event.is_set():
            try:
                batch = self._generator.forward()
                self._queue.put(batch)
            except KeyboardInterrupt:
                self._stop_event.set()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_processes'] = None
        return state

    def __len__(self):
        return len(self._generator)

    def stop(self):
        if self.is_running():
            self._stop_event.set()

        for worker in self._processes:
            if worker.is_alive():
                os.kill(worker.pid, signal.SIGINT)
                worker.join()

        if self._queue is not None:
            self._queue.close()

        self._queue = None
        self._stop_event = None
        self._processes = []

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def forward(self):
        while self.is_running():
            yield self._queue.get()
