import os
import json
import torch
from torch import nn
from model.grud_model import GRUD
from model.multi_model import MultiModel
from model.custom_cross_entropy import CustomCrossEntropyLoss

from data.dataset import RegionDataset
from torch.utils.data import DataLoader
from utils.argument_parser import get_argument
from utils.plot_chart import draw_MAF_R2
from utils.imputation import evaluation, get_device, write_dosage, write_gen, write_output_Oxford_format
# torch.manual_seed(42)
SINGLE_MODEL = ['Higher', 'Lower']
MULTI_MODEL = ['Hybrid']


def run(dataloader, dataset, imp_site_info_list, model_config, args, region):
    device = get_device(args.gpu)
    model_dir = args.model_dir
    result_gen_dir = args.result_gen_dir
    chromosome = args.chromosome
    model_config['device'] = device
    type_model = args.type_model
    if type_model in ['Lower', 'Higher']:
        gamma = args.gamma if type_model == 'Higher' else -args.gamma

    #Init Model
    model = GRUD(model_config, device).to(device)

    loss_fn = CustomCrossEntropyLoss()
    loss_fct = nn.BCEWithLogitsLoss()
    loss = {
        'CustomCrossEntropy': loss_fn, 
        'BCEWithLogitsLoss': loss_fct
    }
    if args.best_model:
        loaded_model = torch.load(os.path.join(model_dir, f'Best_grud_region_{region}.pt'),map_location=torch.device(device))
    else:
        loaded_model = torch.load(os.path.join(model_dir, f'grud_region_{region}.pt'),map_location=torch.device(device))
    model.load_state_dict(loaded_model)
    print(f"Loaded grud_{region} model")
    test_loss, _r2_score, (predictions, labels, dosage) = evaluation(dataloader, model, device, loss)
    print(f"[Evaluate] Loss: {test_loss} \t R2 Score: {_r2_score}")
    write_output_Oxford_format(dosage, imp_site_info_list, chromosome, region, result_gen_dir)
def main():
    args = get_argument()
    root_dir = args.root_dir
    model_config_dir = args.model_config_dir
    batch_size = args.batch_size
    chromosome = args.chromosome
    regions = args.regions.split("-")
    # with open(os.path.join(root_dir, 'index.txt'),'w+') as index_file:
    #     index_file.write("0")
    index_region = args.regions + "_GRUD"

    with open(os.path.join(root_dir, f'{index_region}.txt'), 'w+') as index_file:
        index_file.write("0")

    for region in range(int(regions[0]), int(regions[-1])+1):
        print(f"----------Testing Region {region}----------")
        with open(os.path.join(model_config_dir, f'region_{region}_config.json'), "r") as json_config:
            model_config = json.load(json_config)
            model_config['region'] = region
            model_config['type_model'] = args.type_model

        dataset = RegionDataset(root_dir, index_region, region, chromosome, dataset=args.dataset)
        testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        # print(dataset.site_info_list)
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

if __name__ == "__main__":
    main()
