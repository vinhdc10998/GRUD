import os
import torch
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
        # print(output_prefix)
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

        #     sample_size = len(allele_probs) // 2
        #     values = [0.0] * sample_size
        #     for i in range(sample_size):
        #         h0 = allele_probs[2 * i]
        #         h1 = allele_probs[2 * i + 1]
        #         # print(h0.shape, h1.shape)
        #         if ground_truth == False:
        #             a_a = h0[0] * h1[0]
        #             a_b_b_a = h0[0] * h1[1] + h0[1] * h1[0]
        #             b_b = h0[1] * h1[1]
        #             # print(a_a, a_b_b_a, b_b)
        #             values[i] = (0*a_a + 1*a_b_b_a + 2*b_b).item()

        #         else:
        #             values[i] = (h0 + h1).item()
        #         # print(values[i])
        #         # break
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
        output_prefix = os.path.join(output_prefix, "dosage", f"ground_truth_{chr}_{region}.dosage")
    else:
        output_prefix = os.path.join(output_prefix, "dosage", f"grud_{chr}_{region}.dosage")
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
