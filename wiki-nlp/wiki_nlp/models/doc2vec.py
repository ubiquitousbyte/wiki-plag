import torch
import torch.nn as nn


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


class Sampler(object):
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
    resample()
        Flushes the sampler and recomputes the probabilities again

    use_word(word: str) -> bool
        Returns true if the sampler determines that the word is unique 
        enough to be used during training. 
    """

    def __init__(self, vocabulary_size: int, sampling_rate: float = 0.001):
        self.vocabulary_size = vocabulary_size
        self.sampling_rate = sampling_rate

    def resample(self):
        pass

    def use_word(self, word: str) -> bool:
        pass


class _Batch(object):
    """
    This class represents a batch of training examples.
    The batch consists of a set of documents, the noise samples to use 
    for the documents during training, as well as the context words to 
    aggregate with each document. 

    Attributes
    ----------
    ctxs : List[List[int]]

    docs : List[int]

    y : List[List[int]]

    Methods
    -------
    torchify()
        Convertes the attributes to tensors

    cudify()
        Converts the attributes to CUDA vectors
    """

    def __init__(self):
        self.ctxs = []
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
        self.ctxs = self.ctxs.cuda()
        self.docs = self.docs.cuda()
        self.y = self.y.cuda()
