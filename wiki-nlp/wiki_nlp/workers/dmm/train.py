if __name__ == '__main__':
    from dmm.db import MongoDocumentStore
    from dmm.text import TextProcessor
    from dmm.model import DMMState, DMM
    from dmm.dataset import Vocabulary, Dataset

    store = MongoDocumentStore(
        "mongodb://localhost:27017/wikiplag")
    docs = list(store.read_docs(2))

    text_processor = TextProcessor()

    data = []
    for example in docs:
        tokens = list(text_processor.tokenize(example.text, use_lemmas=True))
        if len(tokens) > 0:
            data.append(tokens)

    dataset = Dataset(data)
    vocab = Vocabulary()
    vocab.build([t for tokens in data for t in tokens], min_freq=1)

    model = DMM(vocab=vocab, n_docs=len(data), dim=10)
    state = model.fit(dataset, epochs=5)
