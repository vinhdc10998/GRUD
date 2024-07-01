import os
import json
import torch
import datetime
from torch import nn
from model.grud_model import GRUD
from model.custom_cross_entropy import CustomCrossEntropyLoss

from data.dataset_inference import RegionInferenceDataset
from torch.utils.data import DataLoader
from utils.argument_parser import get_argument
from utils.imputation import inference, get_device, write_gen, write_output_Oxford_format, oxford_2_vcf
# torch.manual_seed(42)

def run(dataloader, imp_site_info_list, model_config, args, region, device):
    model_dir = args.model_dir
    result_gen_dir = args.result_gen_dir
    chromosome = args.chromosome
    model_config['device'] = device

    #Init Model
    model = GRUD(model_config, device).to(device)

    if args.best_model:
        loaded_model = torch.load(os.path.join(model_dir, f'Best_grud_region_{region}.pt'),map_location=torch.device(device))
    else:
        loaded_model = torch.load(os.path.join(model_dir, f'grud_region_{region}.pt'),map_location=torch.device(device))
    model.load_state_dict(loaded_model)
    predictions, dosage = inference(dataloader, model, device)
    write_output_Oxford_format(dosage, imp_site_info_list, chromosome, region, result_gen_dir)
    write_gen(predictions, imp_site_info_list, chromosome, region, result_gen_dir)
    print(f"[GRUD - Genotype Imputation] Region {region}: {imp_site_info_list[0].position}-{imp_site_info_list[-1].position}")

def main():
    args = get_argument()
    root_dir = args.root_dir
    model_config_dir = args.model_config_dir
    batch_size = args.batch_size
    chromosome = args.chromosome
    regions = args.regions.split("-")
    now = datetime.datetime.now()
    print(f"[GRUD - Genotype Imputation] {now.strftime('%m-%d-%Y %H:%M:%S')}")
    print(f"[GRUD - Genotype Imputation] Input: {root_dir}")
    print(f"[GRUD - Genotype Imputation] Output: {args.result_gen_dir}")
    print(f"[GRUD - Genotype Imputation] Model config: {model_config_dir}")
    print(f"[GRUD - Genotype Imputation] Chromosome: {chromosome}; Batch size: {batch_size}, Region: {range(int(regions[0]), int(regions[-1])+1)}")


    device = get_device(args.gpu)
    for region in range(int(regions[0]), int(regions[-1])+1):
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
            device
        )
    oxford_2_vcf(os.path.join(args.result_gen_dir, 'gen'), args.result_gen_dir, args.sample, chromosome)
    print(f"[GRUD - Genotype Imputation] Output: {os.path.join(args.result_gen_dir, f'gen_{chromosome}.vcf.gz')}")

if __name__ == "__main__":
    main()
