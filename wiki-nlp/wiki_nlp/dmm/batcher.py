import os
import signal
from math import ceil
from typing import (
    Callable,
    Iterable,
    Optional
)

import torch
import torch.multiprocessing as smp
from wiki_nlp.dmm.dataset import Dataset


class BatchGenerator:
    """
    A concurrent batch generator that constructs document batches to feed
    in a Doc2Vec model.

    Attributes
    ----------
    dataset : Dataset
        The dataset of documents to yield batches from
    batch_size : int
        The number of samples to put in a single batch
    ctx_size : int
        The relative context size of a center word.
        The size is relative because it only denotes the number of context
        words on ONE side of a center word.
        Therefore, total context size is actually 2*ctx_size
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

    def __init__(self, dataset: Dataset, batch_size: int, ctx_size: int,
                 n_workers: int = smp.cpu_count()):
        self.n_workers = n_workers
        self._generator = _Generator(dataset, batch_size, ctx_size)

        self._queue = None
        self._stop_event = None
        self._workers = []

    def start(self):
        self._queue = smp.Queue()
        self._stop_event = smp.Event()
        self._spawn_workers()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_workers'] = None
        return state

    def _generate(self):
        while not self._stop_event.is_set():
            try:
                batch = self._generator.next()
                self._queue.put(batch)
            except KeyboardInterrupt:
                self._stop_event.set()

    def __len__(self):
        return len(self._generator)

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def _spawn_workers(self):
        for _ in range(self.n_workers):
            worker = smp.Process(target=self._generate)
            worker.daemon = True
            self._workers.append(worker)
            worker.start()

    def _kill_workers(self):
        for worker in self._workers:
            if worker.is_alive():
                os.kill(worker.pid, signal.SIGINT)
                worker.join()
        self._workers = []

    def stop(self):
        if self.is_running():
            self._stop_event.set()

        self._kill_workers()

        if self._queue is not None:
            self._queue.close()

        self._queue = None
        self._stop_event = None

    def next(self):
        while self.is_running():
            yield self._queue.get()

    def vocab_size(self) -> int:
        return self._generator.vocab_size()


class _Generator:
    """
    A synchronous batch generator that yields batches of inputs to train
    a Doc2Vec model.

    Attributes
    ----------
    dataset: Dataset
        The dataset of documents to yield batches from
    batch_size: int
        The number of samples to put in a single batch
    ctx_size: int
        The relative context size of a center word.
        The size is relative because it only denotes the number of context
        words on ONE side of a center word.
        Therefore, total context size is actually 2*ctx_size

    Methods
    -------
    next() -> _Batch:
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

    def __init__(self, dataset: Dataset, batch_size: int, ctx_size: int):
        """
        Parameters

        dataset: Dataset
            The dataset of documents to yield batches from
        batch_size: int
            The number of samples to put in a single batch
        ctx_size: int
            The relative context size of a center word.
            The size is relative because it only denotes the number of context
            words on ONE side of a center word.
            Therefore, total context size is actually 2*ctx_size
        """

        self.dataset = dataset
        self.batch_size = batch_size
        self.ctx_size = ctx_size
        self.vocab = dataset.vocab
        self.word_sampler = self.vocab.word_sampler
        self.noise_sampler = self.vocab.noise_sampler
        self._state = _GeneratorState(ctx_size)

    def __len__(self):
        num_examples = sum(self.example_counter(d) for d in self.dataset)
        return ceil(num_examples / self.batch_size)

    def next(self) -> '_Batch':
        """Generates a batch and returns it to the caller"""

        doc, word = self._state.forward(self.dataset, self.batch_size,
                                        self.example_counter)

        batch = _Batch(self.ctx_size)

        while len(batch) < self.batch_size:
            if doc == len(self.dataset):
                # All documents have been processed
                # Return the batch
                break

            # Compute the remaining number of context words
            # in the current document
            remaining = len(self.dataset[doc]) - 1 - self.ctx_size
            if word <= remaining:
                # Extract the index of the current word in the vocabulary
                vi = self.vocab[self.dataset[doc][word]]
                if vi < len(self.vocab):
                    if self.vocab.itos(vi) in self.word_sampler:
                        # The word sampler has determined that the
                        # current word is unique enough to be included
                        # in the batch
                        self._fill_batch(batch, doc, word)

                # Advance to the next word
                word += 1
            else:
                # All context words for this document have been processed
                # We therefore advance to the next document
                doc += 1
                word = self.ctx_size

        # The batch has been filled
        # Torchify the batch and return it
        batch.torch_()
        return batch

    def vocab_size(self) -> int:
        return len(self.vocab)

    def _fill_batch(self, batch: '_Batch', doc: int, word: int):
        # Add the document to the batch
        batch.docs.append(doc)

        # Sample from the noise distribution
        noise = self.noise_sampler.sample()

        d = self.dataset[doc]

        # Add the target word to the noise as the first index
        noise.insert(0, self.vocab[d[word]])
        # Add the noise and target word to the batch
        batch.targets.append(noise)

        if self.ctx_size == 0:
            return

        # Construct the context
        ctx = []
        ctx_indices = (word + diff for diff in
                       range(-self.ctx_size, self.ctx_size+1)
                       if diff != 0)
        for i in ctx_indices:
            index = self.vocab[d[i]]
            ctx.append(index)
        batch.ctxs.append(ctx)

    def example_counter(self, doc: Iterable[str], word: int = None):
        if word is not None:
            if len(doc) - word >= self.ctx_size + 1:
                return len(doc) - word - self.ctx_size
            return 0

        if len(doc) >= 2 * self.ctx_size + 1:
            return len(doc) - 2 * self.ctx_size
        return 0


"""
A function pointer that, given a document(a list of tokens), counts
the number of words that are still to be processed relative to the
index of the current center word.

The current center word need not be specified. If it is not, then
all the examples returned equal the length of the list minus total
the context size.
"""
_ExampleCounter = Callable[[Iterable[str], Optional[int]], int]


class _GeneratorState:
    """
    This class represents the state of a batch while it gets constructed
    from a vocabulary and dataset.

    The batch state consists of the current document being processed,
    the current word in the document being processed and a context size
    to offset words.

    Methods
    -------
    forward(dataset: Dataset, batch_size: int, counter: _ExampleCounter) -> Tuple[int, int]
        Returns the current document and word being processed and advances
        the batch state to the next word/document.
        The example counter is used to delimit words and documents.
        The batch size defines the number of documents to put in a single batch.
        The dataset holds the documents to use while constructing the batch
    """

    def __init__(self, ctx_size: int):
        self.ctx_size = ctx_size
        self._doc = smp.RawValue('i', 0)
        self._word = smp.RawValue('i', ctx_size)
        self._lock = smp.Lock()

    def forward(self, dataset: Dataset, batch_size: int, counter: _ExampleCounter):
        """
        This function is thread-safe.

        Parameters
        ----------
        dataset: Dataset
            The set of documents to construct the batch from
        batch_size: int
            The number of documents to put in a single batch
        counter: ExampleCounter
            Counts the number of examples in a document and uses it
            to delimit words and documents in the batch
        """

        with self._lock:
            doc, word = self._doc.value, self._word.value
            self._forward(dataset, batch_size, counter)
            return doc, word

    def _forward(self, dataset: Dataset, batch_size: int, counter: _ExampleCounter):
        """
        This function is NOT thread-safe.

        Parameters
        ----------
        dataset: Dataset
            The set of documents to construct the batch from
        batch_size: int
            The number of documents to put in a single batch
        e: ExampleCounter
            Counts the number of examples in a document and uses it
            to delimit words and documents in the batch
        """

        num_examples = counter(dataset[self._doc.value], self._word.value)

        if num_examples > batch_size:
            # Successfully processed a word batch
            # Position the current word index to the start
            # of the next word batch in the current document and return
            self._word.value += batch_size
            return

        if num_examples == batch_size:
            # Successfully processed a document batch
            if self._doc.value < len(dataset) - 1:
                # If there are more documents in the dataset
                # advance the pointer to the next document
                self._doc.value += 1
            else:
                # If there aren't any more documents
                # position the pointer to the start of the dataset
                self._doc.value = 0

            # Reset the word pointer to the context size
            self._word.value = self.ctx_size
            return

        while num_examples < batch_size:
            # We haven't accumulated enough examples
            # to fit in the batch

            if self._doc.value == len(dataset) - 1:
                # If the current document pointer points
                # to the last document, reset the pointer
                # to the start of the dataset
                self._doc.value = 0
                self._word.value = self.ctx_size
                return

            # The current document had an insufficient number of words
            # to populate the batch.
            # Advance to the next document
            self._doc.value += 1

            # Accumulate examples from the next document
            num_examples += counter(dataset[self._doc.value])

        # Set the word index to the start of the next batch
        self._word.value = (len(dataset[self._doc.value])
                            - self.ctx_size
                            - (num_examples - batch_size))


class _Batch:
    """
    This class represents a batch of training examples.
    The batch consists of a set of documents, the noise samples to use
    for the documents during training, as well as the context words to
    aggregate with each document.

    Attributes
    ----------
    ctxs: List[List[int]]
        A set of context words for each document in the batch
        The outer list represents a mapping between docs and their context
        words.
        In other words, the invariant len(ctxs) = len(docs) always holds.
    docs: List[int]
        A set of documents that are a part of the batch
    targets: List[List[int]]
        The noise samples and the target word for each document in the batch
        The outer list represents a mapping between docs, the target word
        for each doc, and the noise samples for each word.
        Similarly to ctxs, the invariant len(y) = len(docs) always holds.
    Methods
    -------
    torch_()
        Convertes the attributes to tensors

    cuda_()
        Converts the attributes to CUDA vectors
    """

    def __init__(self, ctx_size: int):
        """
        Parameters
        ----------
        ctx_size : int
            The relative context size of a center word.
            The size is relative because it only denotes the number of context
            words on ONE side of a center word.
            Therefore, total context size is actually 2*ctx_size
        """

        self.ctxs = [] if ctx_size > 0 else None
        self.docs = []
        self.targets = []

    def __len__(self):
        return len(self.docs)

    def torch_(self):
        if self.ctxs is not None:
            self.ctxs = torch.LongTensor(self.ctxs)
        self.docs = torch.LongTensor(self.docs)
        self.targets = torch.LongTensor(self.targets)

    def cuda_(self):
        if self.ctxs is not None:
            self.ctxs = self.ctxs.cuda()
        self.docs = self.docs.cuda()
        self.targets = self.targets.cuda()
