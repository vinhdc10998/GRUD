import os
import json
import torch
from torch import nn
from model.grud_model import GRUD
from model.custom_cross_entropy import CustomCrossEntropyLoss

from data.dataset_inference import RegionInferenceDataset
from torch.utils.data import DataLoader
from utils.argument_parser import get_argument
from utils.imputation import inference, get_device, write_gen, write_output_Oxford_format, oxford_2_vcf
# torch.manual_seed(42)

def run(dataloader, imp_site_info_list, model_config, args, region):
    device = get_device(args.gpu)
    model_dir = args.model_dir
    result_gen_dir = args.result_gen_dir
    chromosome = args.chromosome
    model_config['device'] = device
    type_model = args.type_model

    #Init Model
    model = GRUD(model_config, device).to(device)

    if args.best_model:
        loaded_model = torch.load(os.path.join(model_dir, f'Best_grud_region_{region}.pt'),map_location=torch.device(device))
    else:
        loaded_model = torch.load(os.path.join(model_dir, f'grud_region_{region}.pt'),map_location=torch.device(device))
    model.load_state_dict(loaded_model)
    print(f"Loaded grud_{region} model")
    predictions, dosage = inference(dataloader, model, device)
    write_output_Oxford_format(dosage, imp_site_info_list, chromosome, region, result_gen_dir)
    write_gen(predictions, imp_site_info_list, chromosome, region, result_gen_dir)

def main():
    args = get_argument()
    root_dir = args.root_dir
    model_config_dir = args.model_config_dir
    batch_size = args.batch_size
    chromosome = args.chromosome
    regions = args.regions.split("-")
    index_region = args.regions + "_GRUD"

    with open(os.path.join(root_dir, f'{index_region}.txt'), 'w+') as index_file:
        index_file.write("0")

    for region in range(int(regions[0]), int(regions[-1])+1):
        print(f"----------Testing Region {region}----------")
        with open(os.path.join(model_config_dir, f'region_{region}_config.json'), "r") as json_config:
            model_config = json.load(json_config)
            model_config['region'] = region
            model_config['type_model'] = args.type_model

        dataset = RegionInferenceDataset(root_dir, region, chromosome)
        inferenceloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        imp_site_info_list = [
            site_info
            for site_info in dataset.site_info_list if not site_info.array_marker_flag
        ]
        run(
            inferenceloader,
            imp_site_info_list,
            model_config,
            args, 
            region,
        )

    print("----------Imputation Done----------")
    print(f"Writing to gen_{chromosome}.vcf.gz file----------")
    oxford_2_vcf(os.path.join(args.result_gen_dir, 'gen'), args.result_gen_dir, args.sample, chromosome)
    print("----------Writing to VCF Done----------")

if __name__ == "__main__":
    main()
