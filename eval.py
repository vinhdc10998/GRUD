import os
import json
import torch
import numpy as np
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

def run(dataloader, dataset, imp_site_info_list, model_config, args, region):
    device = imputation._get_device(args.gpu)
    type_model = args.model_type
    model_dir = args.model_dir
    result_gen_dir = args.result_gen_dir
    chromosome = args.chromosome
    a1_freq_list = dataset.a1_freq_list
    #Init Model
    model = HybridModel(model_config, a1_freq_list, device, type_model=type_model).float().to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, f'{type_model}_region_{region}.pt')))
    print(f"Loaded {type_model}_{region} model")
    r2_test, predictions, labels = evaluation(dataloader, model, device)
    imputation._write_gen(predictions, imp_site_info_list, chromosome, region, type_model, result_gen_dir)
    plot_chart._draw_MAF_R2(predictions, labels, a1_freq_list, type_model, region, bins=20)
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
    
    check_gen = False
    gen_file = os.path.join(args.result_gen_dir, f"{args.model_type}_{chromosome}.gen")
    if os.path.exists(gen_file) and check_gen:
        label_haplotype = []
        predictions = []
        a1_freq_list = []
        for region in range(int(regions[0]), int(regions[-1])+1):
            print(f"----------Get True Label Region {region}----------")
            dataset = RegionDataset(root_dir, region, chromosome)
            label_haplotype.append(dataset.label_haplotype_list)
            a1_freq_list.append(dataset.a1_freq_list)
        label_haplotype = np.concatenate(label_haplotype, axis=1)
        a1_freq_list = np.concatenate(a1_freq_list)

        with open(gen_file, 'r') as fp:
            for line in fp:
                items = line.rstrip().split()
                hap = items[5:]
                predictions.append(hap)
        predictions = np.array(predictions, dtype=np.int).T
        print(label_haplotype.shape, predictions.shape)
        _r2_score = r2_score(
            label_haplotype,
            predictions
        )
        plot_chart._draw_MAF_R2(torch.tensor(predictions), torch.tensor(label_haplotype), a1_freq_list, args.model_type, args.regions, bins=20)
        print("Evalutate R2 score:", _r2_score)
    else:
        for region in range(int(regions[0]), int(regions[-1])+1):
            print(f"----------Testing Region {region}----------")
            with open(os.path.join(model_config_dir, f'region_{region}_config.json'), "r") as json_config:
                model_config = json.load(json_config)
                model_config['region'] = region
            dataset = RegionDataset(root_dir, region, chromosome)
            testloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            imp_site_info_list = [
                site_info
                for site_info in dataset.site_info_list if not site_info.array_marker_flag
            ]
            run(
                testloader,
                dataset,
                imp_site_info_list,
                model_config,
                args, 
                region,
            )
        
        imputation._merge_gen(
            args.result_gen_dir,
            args.model_type,
            args.chromosome
        )

if __name__ == "__main__":
    main()
