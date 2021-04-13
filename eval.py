import os
import json
import torch
import datetime
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import r2_score
from model.hybrid_model import HybridModel
from data.dataset import RegionDataset
from torch import nn
from torch.utils.data import DataLoader
torch.manual_seed(42)

def draw_chart(train_loss, train_r2_score, val_loss, val_r2_score, region):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f'Region {region}')
    axs[0,0].set_title("Training Loss")
    axs[0,0].plot(train_loss)
    axs[1,0].set_title("Training R2 score")
    axs[1,0].plot(train_r2_score)
    axs[0,1].set_title("Validation Loss")
    axs[0,1].plot(val_loss)
    axs[1,1].set_title("Validation R2 score")
    axs[1,1].plot(val_r2_score)
    fig.tight_layout()
    plt.savefig(f"images/region_{region}.png")

def evaluation(dataloader, model, device):
    #TODO
    '''
        Evaluate model with R square score
    '''
    model.eval()
    _r2_score = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            _, prediction = model(X.float())
            y_pred = torch.argmax(prediction, dim=-1)
            _r2_score += r2_score(
                y.cpu().detach().numpy(),
                y_pred.cpu().detach().numpy()
            )
    _r2_score /= batch+1
    return _r2_score

def run(dataloader, a1_freq_list, model_config, args, region, batch_size=1):
    if args.gpu == True:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == 'cpu': 
            print("You don't have GPU to impute genotype, so I will run by CPU")
        else:
            print(f"You're using GPU {torch.cuda.get_device_name(0)} to impute genotype")
    else: 
        device = 'cpu'
        print("You're using CPU to impute genotype")
    model_type = args.model_type
    model_dir = args.model_dir

    a1_freq_list = torch.tensor(a1_freq_list.tolist()*batch_size)
    #Init Model
    model = HybridModel(model_config, a1_freq_list, batch_size=batch_size, mode=model_type).float().to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, f'model_region_{region}.pt')))
    print(f"Loaded {model_type} model")
    r2_test = evaluation(dataloader, model, device)
    print("EValutate R2 score:", r2_test)

def main():
    description = 'Genotype Imputation'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--root-dir', type=str, required=True,
                        dest='root_dir', help='Data folder')
    parser.add_argument('--model-config-dir', type=str, required=True,
                        dest='model_config_dir', help='Model config folder')
    parser.add_argument('--model-type', type=str, required=True,
                        dest='model_type', help='Model type')
    parser.add_argument('--gpu', type=bool, default=False, required=False,
                        dest='gpu', help='Using GPU')
    parser.add_argument('--batch-size', type=int, default=2, required=False,
                        dest='batch_size', help='Batch size')
    parser.add_argument('--regions', type=str, default=1, required=False,
                        dest='regions', help='Region range')
    parser.add_argument('--chr', type=str, default='chr22', required=False,
                        dest='chromosome', help='Chromosome')
    parser.add_argument('--model-dir', type=str, default='model/weights', required=True,
                        dest='model_dir', help='Weights model dir')
    args = parser.parse_args()

    root_dir = args.root_dir
    model_config_dir = args.model_config_dir
    batch_size = args.batch_size
    chromosome = args.chromosome
    regions = args.regions.split("-")
    with open(os.path.join(root_dir, 'index.txt'),'w+') as index_file:
        index_file.write("0")
    for region in range(int(regions[0]), int(regions[-1])+1):
        print(f"----------Testing Region {region}----------")
        with open(os.path.join(model_config_dir, f'region_{region}_config.json'), "r") as json_config:
            model_config = json.load(json_config)
        dataset = RegionDataset(root_dir, region, chromosome)
        testloader = DataLoader(testloader, batch_size=batch_size, shuffle=True)
        run(
            testloader,
            dataset.a1_freq_list,
            model_config,
            args, region,
            batch_size,
        )

if __name__ == "__main__":
    main()
