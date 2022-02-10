import torch
from wiki_nlp.dmm.model import DMM
from wiki_nlp.dmm.dataset import lemmatize, Dataset
if __name__ == '__main__':
    model = DMM.from_file("dmm_1.pth")
    vocab = torch.load('dmm_vocab_1.pth')
    text = "Bei der Informatik handelt es sich um die Wissenschaft von der systematischen Darstellung, Speicherung, Verarbeitung und Übertragung von Informationen, wobei besonders die automatische Verarbeitung mit Digitalrechnern betrachtet wird."
    proccessed_text = lemmatize([text], n_workers=1)
    dataset = Dataset(documents=proccessed_text, vocab=vocab)

    vector = model.predict(dataset, epochs=20)
