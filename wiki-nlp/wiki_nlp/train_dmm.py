from torchtext.utils import unicode_csv_reader
from torchtext.vocab import vocab as create_vocab
import torch
from collections import Counter, OrderedDict
import spacy
from wiki_nlp.dmm import batcher, dataset, model
import torch.multiprocessing as smp
import pymongo


import argparse

if __name__ == '__main__':
    uri = f"mongodb://wikiplag:wikiplag2021@localhost:27017/wikiplag"
    client = pymongo.MongoClient(uri)

    # Get the collection that holds the input data
    coll = client['wikiplag']["nlp"]

    # Assign each entry a unique mapping to the document matrix of the model
    # for i, entry in enumerate(coll.find()):
    #    coll.update_one({"_id": entry['_id']}, {"$set": {"dmmIndex": i}})

    # Query the paragraphs sorted by their index in the matrix in ascending order
    paragraphs = coll.aggregate([
        {
            "$sort": {"dmmIndex": 1}
        },
        {
            "$project": {
                "text": "$text"
            }
        }
    ])

    # Lemmatize the input
    paragraphs = dataset.lemmatize(
        texts=map(lambda p: p['text'], paragraphs), n_workers=4)
    # Construct the training dataset
    x_train = dataset.Dataset(documents=paragraphs, min_freq=4)

    # Save the vocabulary into the local filesystem
    torch.save(x_train.vocab, "dmm_vocab_1.pth")
    print(f'Vocabulary size: {len(x_train.vocab)}')
    print(f"Beginning training procedure for {len(x_train)} documents.")
    model.DMM.fit(dataset=x_train, dim=100, epochs=20, ctx_size=2, lr=0.025,
                  n_workers=8, save_path="dmm_1.pth")
