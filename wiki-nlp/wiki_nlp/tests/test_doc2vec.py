import unittest

import torch

from wiki_nlp.models import doc2vec
from wiki_nlp.dataset import Vocabulary
from wiki_nlp.text import Token


class DMMTest(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.noise_size = 2
        self.docs = 3
        self.num_words = 15
        self.embedding_size = 10

        self.ctxs = torch.LongTensor([[0, 2, 5, 6], [3, 4, 1, 6]])
        self.doc_indices = torch.LongTensor([1, 2])
        self.y = torch.LongTensor([[1, 3, 4], [2, 4, 7]])
        self.model = doc2vec.DMM(
            self.docs, self.num_words, self.embedding_size)

    def test_forward(self):
        x = self.model.forward(self.ctxs, self.doc_indices, self.y)
        self.assertEqual(x.size()[0], self.batch_size)
        self.assertEqual(x.size()[1], self.noise_size + 1)

    def test_backward(self):
        loss = doc2vec.Loss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        for _ in range(2):
            x = self.model.forward(self.ctxs, self.doc_indices, self.y)
            x = loss.forward(x)
            self.model.zero_grad()
            x.backward()
            optimizer.step()

        self.assertEqual(torch.sum(self.model._D.grad[0, :].data), 0)
        self.assertNotEqual(torch.sum(self.model._D.grad[1, :].data), 0)
        self.assertNotEqual(torch.sum(self.model._D.grad[2, :].data), 0)

        ctxs = self.ctxs.numpy().flatten()
        y = self.y.numpy().flatten()
        for w in range(self.num_words):
            if w in ctxs:
                self.assertNotEqual(
                    torch.sum(self.model._W.grad[w, :].data), 0)
            else:
                self.assertEqual(torch.sum(self.model._W.grad[w, :].data), 0)

            if w in y:
                self.assertNotEqual(
                    torch.sum(self.model._WP.grad[:, w].data), 0)
            else:
                self.assertEqual(torch.sum(self.model._WP.grad[:, w].data), 0)

    def test_loss(self):
        loss = doc2vec.Loss()
        scores = torch.FloatTensor([[12.1, 1.3, 3.5], [18.9, 0.1, 3.4]])
        J = loss.forward(scores)
        self.assertTrue(J.item() >= 0)


class BatchGeneratorTest(unittest.TestCase):

    def setUp(self):
        self.dataset = [[
            Token("bei", "bei", "a"),
            Token("der", "der", "a"),
            Token("informatik", "informatik", "a"),
            Token("handelt", "handelt", "a"),
            Token("es", "es", "a"),
            Token("sich", "sich", "a"),
            Token("um", "um", "a"),
            Token("die", "die", "a"),
            Token("systematischen", "systematischen", "a"),
            Token("darstellung", "darstellung", "a"),
            Token("von", "von", "a"),
            Token("informationen", "informationen", "a"),
        ]]
        self.vocab = Vocabulary()
        self.vocab.build(self.dataset[0], use_lemmas=True, min_freq=1)

    def test_batch_generate(self):
        bs = doc2vec.BatchGenerator(
            vocab=self.vocab,
            dataset=self.dataset,
            batch_size=8,
            ctx_size=2,
            noise_size=4,
            max_size=1,
            workers=1,
            sampling_rate=1
        )
        bs.start()
        batch = next(bs.forward())
        bs.stop()

        self.assertTrue(batch.y.size()[0] <= 8)
        self.assertEqual(batch.y.size()[1], 5)

        self.assertTrue(batch.ctxs.size()[0] <= 8)
        self.assertEqual(batch.ctxs.size()[1], 4)

        self.assertTrue(batch.docs.size()[0] <= 8)


if __name__ == '__main__':
    unittest.main()
