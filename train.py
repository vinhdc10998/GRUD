import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from argparse import ArgumentParser

from model.hybrid_model import HybridModel
from data.dataset import RegionDataset

from torch import nn
from torch.utils.data import DataLoader
torch.manual_seed(42)

def evaluation(prediction, label):
    #TODO
    '''
        Evaluate model with R square score
    '''
    score = None
    return score

def train(dataloader, a1_freq_list, model_config, device,batch_size=1, epochs=200):
    a1_freq_list = torch.tensor(a1_freq_list.tolist()*batch_size)
    
    model = HybridModel(model_config, a1_freq_list, batch_size=batch_size, mode='Higher').float().to(device)
    # print(model)
    print(next(model.parameters()).device)
    loss_fn = model.CustomCrossEntropyLoss
    # loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.01)

    loss_values = []
    for t in range(epochs):
        _r2_score = 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
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
    description = 'Genotype Imputation'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--root-dir', type=str, required=True,
                        dest='root_dir', help='Data folder')
    parser.add_argument('--model-config-dir', type=str, required=True,
                        dest='model_config_dir', help='Model config folder')
    parser.add_argument('--gpu', type=bool, default=False, required=False,
                        dest='gpu', help='Using GPU')
    parser.add_argument('--batch-size', type=int, default=2, required=False,
                        dest='batch_size', help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, required=False,
                        dest='epochs', help='Epochs')
    parser.add_argument('--regions', type=str, default=1, required=False,
                        dest='regions', help='Region range')
    parser.add_argument('--chr', type=str, default='chr22', required=False,
                        dest='chromosome', help='Chromosome')
    args = parser.parse_args()

    root_dir = args.root_dir
    model_config_dir = args.model_config_dir
    batch_size = args.batch_size
    epochs = args.epochs
    chromosome = args.chromosome
    regions = args.regions.split("-")
    
    if args.gpu == True:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == 'cpu': print("You don't have GPU to impute genotype")
    else: 
        device = 'cpu'
        print("You're using CPU to impute genotype")

    for region in range(int(regions[0]), int(regions[-1])+1):
        with open(os.path.join(model_config_dir, f'region_{region}_config.json'), "r") as json_config:
            model_config = json.load(json_config)
        dataset = RegionDataset(root_dir, region, chromosome)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train(dataloader, dataset.a1_freq_list, model_config, device, batch_size, epochs)

if __name__ == "__main__":
    main()