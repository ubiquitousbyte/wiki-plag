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
    def __init__(self, dataset: Dataset, batch_size: int, ctx_size: int):
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
        doc, word = self._state.forward(self.dataset, self.batch_size,
                                        self.example_counter)

        batch = _Batch(self.ctx_size)

        while len(batch) < self.batch_size:
            if doc == len(self.dataset):
                break

            remaining = len(self.dataset[doc]) - 1 - self.ctx_size
            if word <= remaining:
                vi = self.vocab[self.dataset[doc][word]]
                if vi < len(self.vocab):
                    if self.vocab.itos(vi) in self.word_sampler:
                        self._fill_batch(batch, doc, word)
                word += 1
            else:
                doc += 1
                word = self.ctx_size

        batch.torch_()
        return batch

    def vocab_size(self) -> int:
        return len(self.vocab)

    def _fill_batch(self, batch: '_Batch', doc: int, word: int):
        batch.docs.append(doc)

        noise = self.noise_sampler.sample()

        d = self.dataset[doc]

        noise.insert(0, self.vocab[d[word]])
        batch.targets.append(noise)

        if self.ctx_size == 0:
            return

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


_ExampleCounter = Callable[[Iterable[str], Optional[int]], int]


class _GeneratorState:
    def __init__(self, ctx_size: int):
        self.ctx_size = ctx_size
        self._doc = smp.RawValue('i', 0)
        self._word = smp.RawValue('i', ctx_size)
        self._lock = smp.Lock()

    def forward(self, dataset: Dataset, batch_size: int, counter: _ExampleCounter):
        with self._lock:
            doc, word = self._doc.value, self._word.value
            self._forward(dataset, batch_size, counter)
            return doc, word

    def _forward(self, dataset: Dataset, batch_size: int, counter: _ExampleCounter):
        num_examples = counter(dataset[self._doc.value], self._word.value)

        if num_examples > batch_size:
            self._word.value += batch_size
            return

        if num_examples == batch_size:
            if self._doc.value < len(dataset) - 1:
                self._doc.value += 1
            else:
                self._doc.value = 0

            self._word.value = self.ctx_size
            return

        while num_examples < batch_size:
            if self._doc.value == len(dataset) - 1:
                self._doc.value = 0
                self._word.value = self.ctx_size
                return

            self._doc.value += 1
            num_examples += counter(dataset[self._doc.value])

        self._word.value = (len(dataset[self._doc.value])
                            - self.ctx_size
                            - (num_examples - batch_size))


class _Batch:

    def __init__(self, ctx_size: int):
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
