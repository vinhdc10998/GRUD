import json
import torch

from model.hybrid_model import HybridModel
from model.custom_cross_entropy import CustomCrossEntropy
from data.dataset import RegionDataset

from torch.utils.data import DataLoader
from torch import nn

BATCH_SIZE = 1
EPOCHS = 200

def evaluation(prediction, label):
    #TODO
    '''
        Evaluate model with R square score
    '''
    score = None
    return score

def train(dataloader, a1_freq_list, model_config, batch_size=1, epochs=200, gramma=0):
    model = HybridModel(model_config, batch_size=batch_size).float()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    a1_freq_list = torch.tensor(a1_freq_list)
    for t in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction error
            prediction = model(X.float())          
            loss = loss_fn(prediction[0,:].float(), y[0,:,1].long())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.item()
        print(f"[EPOCHS {t}]: loss: {loss:>7f}")
    

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