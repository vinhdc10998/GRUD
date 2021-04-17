import os
import json
import torch
from sklearn.metrics import r2_score
from model.hybrid_model import HybridModel
from data.dataset import RegionDataset
from torch.utils.data import DataLoader
from utils import argument_parser, imputation, plot_chart
torch.manual_seed(42)

def evaluation(dataloader, model, device):
    '''
        Evaluate model with R square score
    '''
    model.eval()
    _r2_score = 0
    with torch.no_grad():
        predictions = []
        labels = []
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            # Compute prediction error
            _, prediction = model(X.float())
            y_pred = torch.argmax(prediction, dim=-1).T
            _r2_score += r2_score(
                y.cpu().detach().numpy(),
                y_pred.cpu().detach().numpy()
            )
            predictions.append(y_pred)
            labels.append(y)
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    _r2_score /= batch+1
    return _r2_score, predictions, labels

def run(dataloader, a1_freq_list, model_config, args, region, batch_size=1):
    device = imputation._get_device(args.gpu)
    type_model = args.model_type
    model_dir = args.model_dir

    #Init Model
    model = HybridModel(model_config, a1_freq_list, device, type_model=type_model).float().to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, f'Best_{type_model}_region_{region}.pt')))
    print(f"Loaded {type_model}_{region} model")
    r2_test, predictions, labels = evaluation(dataloader, model, device)
    plot_chart._draw_MAF_R2(predictions, labels, a1_freq_list, type_model,   region, bins=20)
    print("Evalutate R2 score:", r2_test)

def main():
    args = argument_parser._get_argument()
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
            model_config['region'] = region
        dataset = RegionDataset(root_dir, region, chromosome)
        testloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        run(
            testloader,
            dataset.a1_freq_list,
            model_config,
            args, region,
            batch_size,
        )

if __name__ == "__main__":
    main()
