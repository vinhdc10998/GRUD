import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from model.hybrid_model import HybridModel
from data.dataset import RegionDataset

from torch import nn
from torch.utils.data import DataLoader
torch.manual_seed(42)

BATCH_SIZE = 4
EPOCHS = 200

def evaluation(prediction, label):
    #TODO
    '''
        Evaluate model with R square score
    '''
    score = None
    return score

def train(dataloader, a1_freq_list, model_config, batch_size=1, epochs=200):
    a1_freq_list = torch.tensor(a1_freq_list.tolist()*BATCH_SIZE)
    
    model = HybridModel(model_config, a1_freq_list, batch_size=batch_size, mode='Higher').float()
    print(model)
    loss_fn = model.CustomCrossEntropyLoss
    # loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.01)

    loss_values = []
    for t in range(epochs):
        _r2_score = 0
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction error
            prediction = model(X.float())
            pred = torch.reshape(prediction,(-1,2))
            label = torch.reshape(y[:,:,1], (y.shape[0]*y.shape[1],-1)).long()
            loss = loss_fn(pred, label[:,0])
            _r2_score += r2_score(pred[:,1].detach().numpy(), label[:,0].detach().numpy())
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        loss = loss.item()
        loss_values.append(loss)
        print(f"[EPOCHS {t}]: loss: {loss:>7f}, r2_socre:{_r2_score:>7f}")

    plt.plot(np.array(loss_values), 'r')
    plt.savefig('loss.png')


def main():
    root_dir = 'data/org_data'
    model_config_dir = 'model/config/region_1_config.json'

    with open(model_config_dir, "r") as json_config:
        model_config = json.load(json_config)
    dataset = RegionDataset(root_dir)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    train(dataloader, dataset.a1_freq_list, model_config, BATCH_SIZE, EPOCHS)

if __name__ == "__main__":
    main()