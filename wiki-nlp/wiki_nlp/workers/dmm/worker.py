from typing import Iterable

from celery.exceptions import Ignore
from celery import (
    Celery,
    Task,
    states
)

from dmm.dataset import Dataset
from dmm.text import Token
from dmm.model import (
    DMM,
    DMMState
)


dmm = Celery("dmm")
dmm.config_from_object('dmm.config')


class InferenceTask(Task):
    """
    Base class for an inference task.
    The base class loads a DMM into memory.

    Every worker process will load the model only once and then use it 
    to generate predictions whenever the task queue is populated with documents.
    """

    def __init__(self):
        super(InferenceTask, self).__init__()
        self.model = None

    def __call__(self, *args, **kwargs):
        if not self.model:

            self.state = DMMState.load(self.path)
            n_docs, dim = self.state.model_state['_D'].size()
            self.model = DMM(self.state.vocab, n_docs, dim)
            self.model.load_state_dict(self.state.model_state)
        return self.run(*args, **kwargs)


@dmm.task(bind=True, base=InferenceTask, path='dmm.pth')
def predict(self, data: Iterable[Iterable[Token]]) -> Iterable[Iterable[float]]:
    try:
        vectors = self.model.predict(self.state, Dataset(data))
        return vectors.tolist()
    except Exception as e:
        meta = {
            "exc_type": type(e).__name__,
            "exc_message": repr(e),
            "message": "DMM cannot predict document embeddings"
        }
        self.update_state(state=states.FAILURE, meta=meta)
        raise Ignore()
