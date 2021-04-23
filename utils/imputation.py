import os
import torch
from sklearn.metrics import r2_score
from data.load_data import *

def _calc_loss_r2(x, y, model, loss_fn, r2):
    label = torch.flatten(y.T)
    # Compute prediction error
    logits, prediction = model(x.float())
    loss = loss_fn(logits, label)
    y_pred = torch.argmax(prediction, dim=-1).T
    r2 += r2_score(
        y.cpu().detach().numpy(),
        y_pred.cpu().detach().numpy()
    )
    return loss, r2

def _save_model(model, region, type_model, path, best=False):
    if not os.path.exists(path):
        os.mkdir(path)
    filename = os.path.join(path, f'{type_model}_region_{region}.pt')
    if best == True:
        filename = os.path.join(path, f'Best_{type_model}_region_{region}.pt')
    torch.save(model.state_dict(), filename)

def _get_device(gpu=False):
    if gpu == True:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == 'cpu': 
            print("You don't have GPU to impute genotype, so I will run by CPU")
        else:
            print(f"You're using GPU {torch.cuda.get_device_name(0)} to impute genotype")
    else: 
        device = 'cpu'
        print("You're using CPU to impute genotype")
    return device

def _write_gen(predictions, imp_site_info_list, chr, region, type_model, output_prefix, ground_truth=False):
    output_prefix = os.path.join(output_prefix, f"{type_model}_{chr}_{region}.gen")
    if ground_truth:
        output_prefix = os.path.join(output_prefix, f"{type_model}_{chr}_{region}_GT.gen")

    mkdir(os.path.dirname(output_prefix))
    with open(output_prefix, 'wt') as fp:
        for allele_probs, site_info in zip(predictions.T, imp_site_info_list):
            line = '--- %s %s %s %s ' \
                    % (site_info.id, site_info.position,
                        site_info.a0, site_info.a1)
            if ground_truth:
                a1_freq = site_info.a1_freq
                if site_info.a1_freq > 0.5:
                    a1_freq = 1. - site_info.a1_freq
                    if a1_freq == 0:
                        a1_freq = 0.0001
                line = '--- %s %s %s %s %f ' \
                    % (site_info.id, site_info.position,
                        site_info.a0, site_info.a1, a1_freq)
            
            # alleles = []
            # for allele_index in range(0, len(allele_probs), 2):
            #     alleles.append(allele_probs[allele_index].item() + allele_probs[allele_index+1].item())
            line += ' '.join([str(allele) for allele in allele_probs.tolist()])
            fp.write(line)
            fp.write('\n')

def _merge_gen(folder_dir, type_model, chr, regions):
    print("DEBUG", folder_dir, os.path.join(folder_dir, f"{type_model}_{chr}.gen"))
    gen = os.path.join(folder_dir, f"{type_model}_{chr}.gen")
    mkdir(os.path.dirname(gen))
    with open(gen, 'w+') as mergered_gen:
        for gen in os.listdir(folder_dir):
            gen_tmp = gen.split("_")
            region = gen_tmp[-1].split(".")[0]
            if len(gen_tmp) == 3 and gen_tmp[0] == type_model and region in regions:
                with open(os.path.join(folder_dir, gen), 'r') as genfile:
                    mergered_gen.write(genfile.read())
            
