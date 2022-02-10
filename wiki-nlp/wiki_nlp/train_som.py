from wiki_nlp.dmm.model import DMM
from wiki_nlp.som.model import SOM
import torch
import json
if __name__ == '__main__':
    dmm = DMM.from_file('dmm_1.pth')
    x = dmm._D.detach().clone().data
    x = x.cpu()
    x = (x - torch.mean(x, dim=0)) / torch.std(x, dim=0)
    x.requires_grad = False

    model = SOM(grid=(32, 32, x.size(dim=1)), sigma=1.0,
                learning_rate=0.25, epochs=1)
    model.fit(x, model_path='som_1.pth', map_path='som_map_1')
