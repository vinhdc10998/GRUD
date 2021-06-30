import os
import torch
from sklearn.metrics import r2_score
from data.load_data import mkdir

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
        for (X, y, a1_freq) in dataloader:
            X, y, a1_freq = X.to(device), y.to(device), a1_freq.to(device)
        
            # Compute prediction error
            logit_generator, prediction, logit_discriminator = model(X)
            y_pred = torch.argmax(prediction, dim=-1).T
            test_loss += loss['CustomCrossEntropy'](logit_generator, torch.flatten(y.T), torch.flatten(a1_freq.T)).item() 
            

            label_discriminator = (y_pred != y).float()
            test_loss += loss['BCEWithLogitsLoss'](logit_discriminator.T, label_discriminator).item()

            predictions.append(y_pred)
            labels.append(y)

    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    test_loss /= size
    n_samples = len(labels)
    _r2_score = sum([r2_score(labels[i].cpu().detach().numpy(), predictions[i].cpu().detach().numpy()) for i in range(n_samples)])/n_samples
    return test_loss, _r2_score, (predictions, labels)

def train(dataloader, model, device, loss, optimizer, scheduler):
    '''
        Train model GRU
    '''
    model.train()
    _r2_score = 0
    train_loss = 0
    predictions = []
    labels = []

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
        loss_BCE = loss['BCEWithLogitsLoss'](logit_discriminator.T, label_discriminator)

        total_loss = loss_BCE + loss_crossentropy
        predictions.append(y_pred)
        labels.append(y)
        #Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_loss = total_loss.item()

    scheduler.step()
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)

    n_samples = len(labels)
    _r2_score = sum([r2_score(labels[i].cpu().detach().numpy(), predictions[i].cpu().detach().numpy()) for i in range(n_samples)])/n_samples

    return train_loss, _r2_score

def save_model(model, region, type_model, path, best=False):
    if not os.path.exists(path):
        os.mkdir(path)
    filename = os.path.join(path, f'{type_model}_region_{region}.pt')
    if best == True:
        filename = os.path.join(path, f'Best_{type_model}_region_{region}.pt')
    torch.save(model.state_dict(), filename)

def save_check_point(model, optimizer, epochs, region, type_model, path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        os.remove(os.path.join(path, os.listdir(path)[-1]))
    filename = os.path.join(path, f'{type_model}_region_{region}_checkpoint_{epochs}.pt')
    torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, filename)

def get_device(gpu=False):
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

def write_gen(predictions, imp_site_info_list, chr, region, type_model, output_prefix, ground_truth=False):
    if ground_truth:
        output_prefix = os.path.join(output_prefix, f"ground_truth_{chr}_{region}.gen")
        # print(output_prefix)
    else:
        output_prefix = os.path.join(output_prefix, f"{type_model}_{chr}_{region}.gen")


    mkdir(os.path.dirname(output_prefix))
    with open(output_prefix, 'wt') as fp:
        # print(predictions.T.shape,  len(imp_site_info_list))
        for allele_probs, site_info in zip(predictions.T, imp_site_info_list):
            # print("222222222222")
            a1_freq = site_info.a1_freq
            if site_info.a1_freq > 0.5:
                a1_freq = 1. - site_info.a1_freq
                if a1_freq == 0:
                    a1_freq = 0.0001
            line = '--- %s %s %s %s %f ' \
                % (site_info.id, site_info.position,
                    site_info.a0, site_info.a1, a1_freq)
            # print(line)
            # alleles = []
            # for allele_index in range(0, len(allele_probs), 2):
            #     alleles.append(allele_probs[allele_index].item() + allele_probs[allele_index+1].item())
            line += ' '.join([str(allele) for allele in allele_probs.tolist()])
            fp.write(line)
            fp.write('\n')

# def merge_gen(folder_dir, type_model, chr, regions):
#     print("DEBUG", folder_dir, os.path.join(folder_dir, f"{type_model}_{chr}.gen"))
#     gen = os.path.join(folder_dir, f"{type_model}_{chr}.gen")
#     mkdir(os.path.dirname(gen))
#     with open(gen, 'w+') as mergered_gen:
#         for gen in os.listdir(folder_dir):
#             gen_tmp = gen.split("_")
#             region = gen_tmp[-1].split(".")[0]
#             if len(gen_tmp) == 3 and gen_tmp[0] == type_model and region in regions:
#                 with open(os.path.join(folder_dir, gen), 'r') as genfile:
#                     mergered_gen.write(genfile.read())
            
