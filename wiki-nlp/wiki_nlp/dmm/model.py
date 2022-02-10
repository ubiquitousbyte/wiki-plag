from typing import Generator
from sys import stdout

import torch
import torch.nn as nn
from torch.optim import SGD
import torch.multiprocessing as smp
from torch.nn.functional import cosine_similarity


from wiki_nlp.dmm.dataset import Dataset
from wiki_nlp.dmm.batcher import BatchGenerator, _Batch


class DMM(nn.Module):

    def __init__(self, n_words: int, n_docs: int, dim: int = 100):
        super(DMM, self).__init__()

        self._D = nn.Parameter(
            torch.randn(n_docs, dim), requires_grad=True
        )
        self._W = nn.Parameter(
            torch.cat((torch.randn(n_words, dim),
                      torch.zeros(1, dim))),
            requires_grad=True
        )
        self._WP = nn.Parameter(
            torch.cat((torch.randn(dim, n_words),
                      torch.randn(dim, 1)), dim=1),
            requires_grad=True
        )

    def forward(self, ctxs, docs, targets):
        h = torch.add(self._D[docs, :], torch.sum(self._W[ctxs, :], dim=1))
        return torch.bmm(h.unsqueeze(1), self._WP[:, targets].permute(1, 0, 2)).squeeze()

    @staticmethod
    def _print_progress(batch: int, epoch: int, batch_count: int):
        step_progress = round((batch + 1) / batch_count * 100)
        print("\rEpoch {:d}".format(epoch + 1), end='')
        stdout.write(" - {:d}%".format(step_progress))
        stdout.flush()

    @staticmethod
    def fit(dataset: Dataset, dim: int, epochs: int,
            batch_size: int = 32, ctx_size: int = 4,
            lr: float = 0.025, n_workers: int = smp.cpu_count(),
            save_path: str = None):
        bg = BatchGenerator(dataset=dataset, batch_size=batch_size,
                            ctx_size=ctx_size, n_workers=n_workers)
        n_batches = len(bg)
        bg.start()
        try:
            DMM._fit(generator=bg.next(), n_words=bg.vocab_size(),
                     n_docs=len(dataset), dim=dim,
                     n_batches=n_batches, epochs=epochs,
                     lr=lr, save_path=save_path)
        except:
            bg.stop()

    @classmethod
    def from_file(cls, path: str) -> 'DMM':
        model_state = torch.load(path)
        model_state_dict = model_state['model_state_dict']
        n_words = model_state_dict['_W'].size(dim=0)-1
        n_docs = model_state_dict['_D'].size(dim=0)
        dim = model_state_dict['_D'].size(dim=1)
        model = cls(n_words=n_words, n_docs=n_docs, dim=dim)
        model.load_state_dict(model_state_dict)
        return model

    def predict(self, dataset: Dataset, epochs: int, batch_size: int = 32,
                ctx_size: int = 2, lr: float = 0.025, n_workers: int = 2):

        d = torch.zeros(len(dataset), self._W.size(dim=1))

        with torch.no_grad():
            self._D.data = torch.cat((d, self._D.data))

        self._W.requires_grad = False
        self._WP.requires_grad = False

        bg = BatchGenerator(dataset=dataset, batch_size=batch_size,
                            ctx_size=ctx_size, n_workers=n_workers)
        n_batches = len(bg)
        bg.start()
        try:
            DMM._fit_existing_dmm(self, bg.next(), n_batches, epochs, lr)
        except:
            bg.stop()

        return self._D[:len(dataset), :].detach().clone()

    def most_similar(self, doc: torch.FloatTensor, topn: int = 4):
        with torch.no_grad():
            dists = torch.matmul(self._D, doc.T)
            ms = torch.topk(dists.squeeze(), topn)
            return ms.values, ms.indices
        #dists = cosine_similarity(self._D, doc)
        #ms = torch.topk(dists, topn)
        # return ms.values, ms.indices

    @staticmethod
    def _fit_existing_dmm(model: 'DMM', generator: Generator, n_batches: int,
                          epochs: int, lr: float, save_path: str = None):

        optimizer = SGD(params=model.parameters(), lr=lr)

        if torch.cuda.is_available():
            model.cuda()

        loss_func = Loss()
        current_loss = float("inf")

        for epoch in range(epochs):
            loss = 0
            for batch_index in range(n_batches):
                batch: _Batch = next(generator)
                if torch.cuda.is_available():
                    batch.cuda_()
                x = model.forward(batch.ctxs, batch.docs, batch.targets)
                del batch

                j = loss_func.forward(x)

                loss += j.item()

                model.zero_grad()
                j.backward()
                optimizer.step()
                DMM._print_progress(batch_index, epoch, n_batches)

            loss /= float(n_batches)
            print("\nLoss ", loss)
            is_best_loss = loss < current_loss
            current_loss = min(loss, current_loss)

            if save_path is not None and is_best_loss:
                state = {
                    'epoch': epoch + 1,
                    'lr': lr,
                    'loss': current_loss,
                    "model_state_dict": model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, save_path)

    @staticmethod
    def _fit(generator: Generator, n_words: int, n_docs: int, dim: int,
             n_batches: int, epochs: int, lr: float, save_path: str = None):

        model = DMM(n_words=n_words, n_docs=n_docs, dim=dim)
        DMM._fit_existing_dmm(model, generator, n_batches,
                              epochs, lr, save_path)


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self._f = nn.LogSigmoid()

    def forward(self, scores):
        n = scores.size(dim=0)
        k = scores.size(dim=1) - 1
        return -torch.sum(self._f(scores[:, 0]) +
                          torch.sum(self._f(-scores[:, 1:]), dim=1) / k) / n
