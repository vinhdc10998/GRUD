import os
import json
import torch
from model.single_model import SingleModel
from model.multi_model import MultiModel
from model.custom_cross_entropy import CustomCrossEntropyLoss

from data.dataset import RegionDataset
from torch.utils.data import DataLoader
from utils.argument_parser import get_argument
from utils.plot_chart import draw_MAF_R2
from utils.imputation import evaluation, get_device, write_gen
# torch.manual_seed(42)
SINGLE_MODEL = ['Higher', 'Lower']
MULTI_MODEL = ['Hybrid']


def run(dataloader, dataset, imp_site_info_list, model_config, args, region):
    device = get_device(args.gpu)
    type_model = args.model_type
    model_dir = args.model_dir
    result_gen_dir = args.result_gen_dir
    chromosome = args.chromosome
    gamma = args.gamma if type_model == 'Higher' else -args.gamma


    #Init Model
    if type_model in SINGLE_MODEL:
        gamma = args.gamma if type_model == 'Higher' else -args.gamma
        model = SingleModel(model_config, device, type_model=type_model).to(device) 

    elif type_model in MULTI_MODEL:
        gamma = 0
        if args.best_model: 
            model_config['lower_path'] = os.path.join(args.model_dir, f'Best_Lower_region_{region}.pt')
            model_config['higher_path'] = os.path.join(args.model_dir, f'Best_Higher_region_{region}.pt')
        else:
            model_config['lower_path'] = os.path.join(args.model_dir, f'Lower_region_{region}.pt')
            model_config['higher_path'] = os.path.join(args.model_dir, f'Higher_region_{region}.pt')
        
        model = MultiModel(model_config, device, type_model=type_model).to(device)
    # model = SingleModel(model_config, device, type_model=type_model).to(device)

    #Init Model
    loss_fn = CustomCrossEntropyLoss(gamma)
    if args.best_model:
        loaded_model = torch.load(os.path.join(model_dir, f'Best_{type_model}_region_{region}.pt'),map_location=torch.device(device))
    else:
        loaded_model = torch.load(os.path.join(model_dir, f'{type_model}_region_{region}.pt'),map_location=torch.device(device))
    model.load_state_dict(loaded_model)
    print(f"Loaded {type_model}_{region} model")
    test_loss, _r2_score, (predictions, labels) = evaluation(dataloader, model, device, loss_fn)
    print(predictions.shape, labels.shape)
    print(f"[Evaluate] Loss: {test_loss} \t R2 Score: {_r2_score}")
    write_gen(predictions, imp_site_info_list, chromosome, region, type_model, result_gen_dir)
    write_gen(labels, imp_site_info_list, chromosome, region, type_model, result_gen_dir, ground_truth=True)
    
    # draw_MAF_R2(predictions, labels, a1_freq_list, type_model, region, bins=30)

def main():
    args = get_argument()
    root_dir = args.root_dir
    model_config_dir = args.model_config_dir
    batch_size = args.batch_size
    chromosome = args.chromosome
    regions = args.regions.split("-")
    # with open(os.path.join(root_dir, 'index.txt'),'w+') as index_file:
    #     index_file.write("0")
    
    for region in range(int(regions[0]), int(regions[-1])+1):
        print(f"----------Testing Region {region}----------")
        with open(os.path.join(model_config_dir, f'region_{region}_config.json'), "r") as json_config:
            model_config = json.load(json_config)
            model_config['region'] = region
        dataset = RegionDataset(root_dir, region, chromosome, dataset=args.dataset)
        testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        # imp_site_info_list = [
        #     site_info
        #     for site_info in dataset.site_info_list if not site_info.array_marker_flag
        # ]
        run(
            testloader,
            dataset,
            dataset.site_info_list,
            model_config,
            args, 
            region,
        )

if __name__ == "__main__":
    main()
