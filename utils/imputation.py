import os
import torch
import pandas as pd
import gzip
import subprocess
import shutil

from sklearn.metrics import matthews_corrcoef
from data.load_data import mkdir
from torch.nn import functional as F
from scipy.stats import pearsonr

def evaluation(dataloader, model, device, loss):
    '''
        Evaluate model with R square score
    '''
    model.eval()  
    size = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        predictions = []
        labels = []
        dosage = []
        for (X, y, a1_freq) in dataloader:
            X, y, a1_freq = X.to(device), y.to(device), a1_freq.to(device)
        
            # Compute prediction error
            logit_generator, prediction, _ = model(X)
            # print(prediction.shape)
            # print(prediction)

            y_pred = torch.argmax(prediction, dim=-1).T
            # print(y_pred.shape)
            # y_pred = prediction[:,:,1].T
            # print(y_pred.shape)
            test_loss += loss['CustomCrossEntropy'](logit_generator, torch.flatten(y.T), torch.flatten(a1_freq.T)).item() 
            # print("DEBUG", prediction.T.shape)

            dosage.append(prediction)
            predictions.append(y_pred)
            labels.append(y)
    
    dosage = torch.cat(dosage, dim=1)
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    test_loss /= size
    _r2_score = pearsonr(labels.cpu().detach().numpy().flatten(), predictions.cpu().detach().numpy().flatten())[0]**2
    return test_loss, _r2_score, (predictions, labels, dosage)

def train(dataloader, model, device, loss, optimizer, scheduler):
    '''
        Train model GRU
    '''
    model.train()
    _r2_score = 0
    train_loss = 0
    predictions = []
    labels = []
    # for name, param in model.named_parameters():
    #     if 'discriminator' in name or 'generator.linear' in name:
    #         print(name, param.grad)

    for batch, (X, y, a1_freq) in enumerate(dataloader):
        X, y, a1_freq = X.to(device), y.to(device), a1_freq.to(device)
        # Compute prediction error
        logit_generator, prediction, logit_discriminator = model(X)
        
        loss_crossentropy = loss['CustomCrossEntropy'](logit_generator, torch.flatten(y.T), torch.flatten(a1_freq.T))
        y_pred = torch.argmax(prediction, dim=-1).T
        
        '''
        Loss discriminator
            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.
        '''
        label_discriminator = (y_pred != y).float()
        loss_BCE = loss['BCEWithLogitsLoss'](logit_discriminator, label_discriminator)

        total_loss = loss_BCE + loss_crossentropy
        predictions.append(y_pred)
        labels.append(y)
        #Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_loss = total_loss.item()

    scheduler.step()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    labels = torch.cat(labels, dim=0).cpu().detach().numpy()
    _r2_score = pearsonr(labels.flatten(), predictions.flatten())[0]**2

    return train_loss, _r2_score

def save_model(model, region, path, type_model="dis", best=False):
    if not os.path.exists(path):
        os.mkdir(path)
    filename = os.path.join(path, f'grud_{type_model}_region_{region}.pt')
    if best == True:
        filename = os.path.join(path, f'Best_grud_{type_model}_region_{region}.pt')
    torch.save(model.state_dict(), filename)

def save_check_point(model, optimizer, epochs, region, path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        os.remove(os.path.join(path, os.listdir(path)[-1]))
    filename = os.path.join(path, f'grud_region_{region}_checkpoint_{epochs}.pt')
    torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, filename)

def get_device(gpu=False):
    if gpu:
        device = f"cuda:{gpu}" 
        print(f"You're using GPU {torch.cuda.get_device_name(int(gpu))} to impute genotype")
    else: 
        device = 'cpu'
        print("You're using CPU to impute genotype")
    return device

def write_dosage(dosage, imp_site_info_list, chr, region, output_prefix, ground_truth=False):
    print(dosage.shape)
    if ground_truth:
        output_prefix = os.path.join(output_prefix, "dosage", f"ground_truth_{chr}_{region}.dosage")
    else:
        output_prefix = os.path.join(output_prefix, "dosage", f"grud_{chr}_{region}.dosage")
    
    mkdir(os.path.dirname(output_prefix))
    with open(output_prefix, 'wt') as fp:
        tmp_evens = dosage[:,0::2]
        tmp_odds = dosage[:,1::2]
        a_a = tmp_evens[:, :, 0] * tmp_odds[:, :, 0]
        a_b_b_a = tmp_evens[:, :, 0] * tmp_odds[:, :, 1] + tmp_evens[:, :, 1] * tmp_odds[:, :, 0]
        b_b = tmp_evens[:, :, 1] * tmp_odds[:, :, 1]
        dosage = (1*a_b_b_a + 2*b_b).cpu().detach().numpy()
        # print(kkk.shape)
        # dosage = kkk.T.to(device)
        print(dosage.shape)
        for allele_probs, site_info in zip(dosage, imp_site_info_list):
        #     # print(allele_probs.shape)
            a1_freq = site_info.a1_freq
            if site_info.a1_freq > 0.5:
                a1_freq = 1. - site_info.a1_freq
                if a1_freq == 0:
                    a1_freq = 0.00001

            line = '--- %s %s %s %s %f ' \
                   % (f'chr22_{site_info.position}_{site_info.a0}_{site_info.a1}', site_info.position,
                      site_info.a0, site_info.a1, a1_freq)
            line += ' '.join(map(str, allele_probs))
            fp.write(line)
            fp.write('\n')
            
def write_gen(predictions, imp_site_info_list, chr, region, output_prefix_t, ground_truth=False):
    if ground_truth:
        output_prefix = os.path.join(output_prefix_t, "gen", f"ground_truth_{chr}_{region}.gen")
    else:
        output_prefix = os.path.join(output_prefix_t, "gen", f"grud_{chr}_{region}.gen")

    mkdir(os.path.dirname(output_prefix))
    with open(output_prefix, 'wt') as fp:
        for allele_probs, site_info in zip(predictions.T, imp_site_info_list):
            a1_freq = site_info.a1_freq
            if site_info.a1_freq > 0.5:
                a1_freq = 1. - site_info.a1_freq
                if a1_freq == 0:
                    a1_freq = 0.0001
            line = '--- %s %s %s %s %f ' \
                % (f'chr22_{site_info.position}_{site_info.a0}_{site_info.a1}', site_info.position,
                    site_info.a0, site_info.a1, a1_freq)
        
            # alleles = []
            # for allele_index in range(0, len(allele_probs), 2):
            #     alleles.append(allele_probs[allele_index].item() + allele_probs[allele_index+1].item())
            line += ' '.join([str(allele) for allele in allele_probs.tolist()])
            fp.write(line)
            fp.write('\n')



def write_output_Oxford_format(dosage, imp_site_info_list, chr, region, output_prefix, ground_truth=False):
    if ground_truth:
        output_prefix = os.path.join(output_prefix, "oxford", f"ground_truth_{chr}_{region}.dosage")
    else:
        output_prefix = os.path.join(output_prefix, "oxford", f"grud_{chr}_{region}.dosage")
    mkdir(os.path.dirname(output_prefix))
    with open(output_prefix, 'wt') as fp:
        tmp_evens = dosage[:,0::2]
        tmp_odds = dosage[:,1::2]
        a_a = (tmp_evens[:, :, 0] * tmp_odds[:, :, 0]).cpu().detach().numpy()
        a_b_b_a = (tmp_evens[:, :, 0] * tmp_odds[:, :, 1] + tmp_evens[:, :, 1] * tmp_odds[:, :, 0]).cpu().detach().numpy()
        b_b = (tmp_evens[:, :, 1] * tmp_odds[:, :, 1]).cpu().detach().numpy()
        sample_size = a_a.shape[1]
        values = [0.0] * 3 * sample_size
        for value1, value2, value3, site_info in zip(a_a, a_b_b_a, b_b, imp_site_info_list):
            a1_freq = site_info.a1_freq
            if site_info.a1_freq > 0.5:
                a1_freq = 1. - site_info.a1_freq
            line = '--- %s %s %s %s ' \
                   % (f'chr22_{site_info.position}_{site_info.a0}_{site_info.a1}', site_info.position,
                      site_info.a0, site_info.a1)
            for i in range(sample_size):
                values[3*i] = value1[i]
                values[3*i + 1] = value2[i]
                values[3*i + 2] = value3[i]
            line += ' '.join(map(str, values))
            fp.write(line)
            fp.write('\n')

def oxford_2_vcf(path_gen, vcf_path, sample_name_path, chr):
    output_path = os.path.join(path_gen, "gen.txt")
    if os.path.exists(output_path):
        os.remove(output_path)
    command_line = f'cat $(ls -v {os.path.join(path_gen,"*")}) >> {output_path}'
    os.system(command_line)
    
    header_path = './header.txt'
    with open(header_path, 'r') as fp:
        header = fp.read()

    df = pd.read_csv(output_path, sep=' ', header=None)
    chrom=chr
    filterValue='.'
    qual='PASS'
    info='.'
    formatValue='GT'
    VCF = header
    sample_header = '\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT'

    with open(sample_name_path, 'r') as fp:
        sample_lsit = fp.read().rstrip().split()

    VCF += sample_header + '\t' + '\t'.join(sample_lsit)
    VCF += '\n'

    for index, row in df.iterrows():
        id = row[1]
        pos= row[2]
        ref=row[3]
        alt=row[4]
        gt=row[6:].tolist()
        info = '.'
        tmp = []
        for indkex in range(0, len(gt)-1, 2):
            tmp.append(str(gt[indkex]) + "|" + str(gt[indkex+1]))
        tmp = "\t".join(tmp)
        content_VCF = f'{chrom}\t{pos}\t{id}\t{ref}\t{alt}\t{qual}\t{filterValue}\t{info}\t{formatValue}\t{tmp}\n'
        VCF += content_VCF

    with gzip.open(os.path.join(vcf_path, f'gen_{chrom}.vcf.gz'), "wb") as fp:
        fp.write(VCF.encode())
    shutil.rmtree(path_gen, ignore_errors=True)

