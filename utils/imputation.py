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
        for (X, y, y_dosage, a1_freq) in dataloader:
            X, y, a1_freq = X.to(device, non_blocking=True), y.to(device, non_blocking=True), a1_freq.to(device, non_blocking=True)
            X = torch.reshape(X, (X.shape[0]*X.shape[1],-1, 2))
            y = torch.reshape(y, (y.shape[0]*y.shape[1],-1))
            # y = torch.stack(y, dim=0)
            a1_freq = torch.cat([a1_freq, a1_freq])
            # Compute prediction error
            logit_generator, prediction, _ = model(X)
            print("dosage", y_dosage.shape)
            y_pred = torch.argmax(prediction, dim=-1).T
            # test_loss += loss['CustomCrossEntropy'](logit_generator, torch.flatten(y.T), torch.flatten(a1_freq.T)).item() 
            test_loss += loss['CustomCrossEntropy'](logit_generator, torch.flatten(y.T)).item() 
            
            dosage.append(prediction)
            predictions.append(y_pred)
            labels.append(y)
    
    dosage = torch.cat(dosage, dim=0)
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

    for batch, (X, y, y_dosage, a1_freq) in enumerate(dataloader):
        X, y, a1_freq, y_dosage = X.to(device, non_blocking=True), y.to(device, non_blocking=True), a1_freq.to(device, non_blocking=True), y_dosage.to(device, non_blocking=True)
        # Compute prediction error
        X = torch.reshape(X, (X.shape[0]*X.shape[1],-1, 2))
        y = torch.reshape(y, (y.shape[0]*y.shape[1],-1))
        a1_freq = torch.cat([a1_freq, a1_freq])
        logit_generator, prediction, logit_discriminator = model(X)
        
        ###get dosage
        tmp_evens = prediction[:,0::2]
        tmp_odds = prediction[:,1::2]
        a_b_b_a = tmp_evens[:, :, 0] * tmp_odds[:, :, 1] + tmp_evens[:, :, 1] * tmp_odds[:, :, 0]
        b_b = tmp_evens[:, :, 1] * tmp_odds[:, :, 1]
        kkk = (1*a_b_b_a + 2*b_b)
        dosage = kkk.T.to(device)

        loss_l1 = loss['L1Loss'](dosage, y_dosage)
        # loss_crossentropy = loss['CustomCrossEntropy'](logit_generator, torch.flatten(y.T), torch.flatten(a1_freq.T))
        loss_crossentropy = loss['CustomCrossEntropy'](logit_generator, torch.flatten(y.T))

        y_pred = torch.argmax(prediction, dim=-1).T
        
        '''
        Loss discriminator
            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.
        '''

        # print(y_pred.shape, y.shape)
        label_discriminator = (y_pred != y).float()
        loss_BCE = loss['BCEWithLogitsLoss'](logit_discriminator, label_discriminator)

        total_loss = loss_BCE + loss_crossentropy + loss_l1
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

def save_model(model, region, path, best=False):
    if not os.path.exists(path):
        os.mkdir(path)
    filename = os.path.join(path, f'grud_region_{region}.pt')
    if best == True:
        filename = os.path.join(path, f'Best_grud_region_{region}.pt')
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

def write_dosage(dosage, imp_site_info_list, chr, region, output_prefix, ground_truth=False):
    if ground_truth:
        output_prefix = os.path.join(output_prefix, "dosage", f"ground_truth_{chr}_{region}.dosage")
        # print(output_prefix)
    else:
        output_prefix = os.path.join(output_prefix, "dosage", f"grud_{chr}_{region}.dosage")
    
    mkdir(os.path.dirname(output_prefix))
    with open(output_prefix, 'wt') as fp:
        tmp_evens = dosage[:,0::2]
        tmp_odds = dosage[:,1::2]
        a_b_b_a = tmp_evens[:, :, 0] * tmp_odds[:, :, 1] + tmp_evens[:, :, 1] * tmp_odds[:, :, 0]
        b_b = tmp_evens[:, :, 1] * tmp_odds[:, :, 1]
        dosage = (1*a_b_b_a + 2*b_b).cpu().detach().numpy()
        # print(kkk.shape)
        # dosage = kkk.T.to(device)
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

def write_gen(predictions, dosage_all, imp_site_info_list, chr, region, output_prefix_t, ground_truth=False):
    if ground_truth:
        output_prefix = os.path.join(output_prefix_t, "gen", f"ground_truth_{chr}_{region}.gen")
        output_prefix_dosage = os.path.join(output_prefix_t, "dosage", f"ground_truth_{chr}_{region}.dosage")
        # print(output_prefix)
    else:
        output_prefix = os.path.join(output_prefix_t, "gen", f"grud_{chr}_{region}.gen")
        output_prefix_dosage = os.path.join(output_prefix_t, "dosage", f"grud_{chr}_{region}.dosage")


    mkdir(os.path.dirname(output_prefix))
    mkdir(os.path.dirname(output_prefix_dosage))


    with open(output_prefix, 'wt') as fp, open(output_prefix_dosage, 'wt') as dosage_file:
        # print(predictions.T.shape,  len(imp_site_info_list))
        for index, (allele_probs, dosage, site_info) in enumerate(zip(predictions.T, dosage_all, imp_site_info_list)):

            a1_freq = site_info.a1_freq
            if site_info.a1_freq > 0.5:
                a1_freq = 1. - site_info.a1_freq
                if a1_freq == 0:
                    a1_freq = 0.0001
            sample_size = len(dosage) // 2
            values = [0.0] * sample_size
            for i in range(sample_size):
                h0 = dosage[2 * i]
                h1 = dosage[2 * i + 1]
                # print(h0.shape, h1.shape)
                if ground_truth == False:
                    a_a = h0[0] * h1[0]
                    a_b_b_a = h0[0] * h1[1] + h0[1] * h1[0]
                    b_b = h0[1] * h1[1]
                    # print(a_a, a_b_b_a, b_b)
                    values[i] = (0*a_a + 1*a_b_b_a + 2*b_b).item()

                else:
                    values[i] = (h0 + h1).item()
            
            
            line = '--- %s %s %s %s %f ' \
                % (f'{chr}_{site_info.position}_{site_info.a0}_{site_info.a1}', site_info.position,
                    site_info.a0, site_info.a1, a1_freq)
            line_dosage = '--- %s %s %s %s %f ' \
                % (f'{chr}_{site_info.position}_{site_info.a0}_{site_info.a1}', site_info.position,
                    site_info.a0, site_info.a1, a1_freq)
            line_dosage += ' '.join(map(str, values))
            line += ' '.join([str(allele) for allele in allele_probs.tolist()])
            fp.write(line)
            fp.write('\n')
            dosage_file.write(line_dosage)
            dosage_file.write('\n')

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


def train_ae(dataloader, encoder, decoder, discriminator, encoder_optimizer, decoder_optimizer, criterion, device):
    encoder.train()
    decoder.train()
    
    _r2_score = 0
    train_loss = 0
    predictions = []
    labels = []
    # for name, param in decoder.named_parameters():
    #     print(name, param.grad)
    #     break

    for batch, (X, y, a1_freq) in enumerate(dataloader):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        X, y, a1_freq = X.to(device), y.to(device), a1_freq.to(device)

        batch_size = X.shape[0]
        encoder_hidden = encoder.init_hidden(batch_size)

        input_length = X.shape[1]
        target_length = y.shape[1]

        loss_crossentropy = 0
        loss_BCE = 0
        encoder_outputs = torch.zeros(batch_size, input_length, 40, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(X[:, ei, :], encoder_hidden)
            encoder_outputs[:, ei] = encoder_output[:, 0]

        decoder_input = torch.zeros((y.shape[0]), dtype=torch.int, device=device)
        decoder_hidden = encoder_hidden

        predict = []
        label = []
        for di in range(target_length):
            decoder_logit, decoder_hidden, fake_decode  = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss_crossentropy += criterion['CrossEntropy'](decoder_logit[0], y[:, di])
            tmp = torch.squeeze(torch.argmax(F.softmax(decoder_logit, dim=-1), dim=-1))
            decoder_input = tmp
            discriminator_logit = discriminator(fake_decode.detach()).T
            label_discriminator = (tmp != y[:, di]).float()
            loss_BCE += criterion['BCEWithLogitsLoss'](torch.squeeze(discriminator_logit), label_discriminator)

            predict.append(decoder_logit[0])
            label.append(y[:, di])
        y_pred = torch.argmax(torch.stack(predict), dim=-1).T
        predictions.append(y_pred)

        labels.append(torch.stack(label).T)
        total_loss = loss_crossentropy + loss_BCE
        total_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        train_loss = total_loss.item()

    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)

    n_samples = len(labels)
    _r2_score = sum([matthews_corrcoef(labels[i].cpu().detach().numpy(), predictions[i].cpu().detach().numpy()) for i in range(n_samples)])/n_samples

    return train_loss/target_length, _r2_score

def eval_ae(dataloader, encoder, decoder, criterion, device):
    encoder.eval()
    decoder.eval()
    _r2_score = 0
    test_loss = 0
    predictions = []
    labels = []
    with torch.no_grad():
        for batch, (X, y, a1_freq) in enumerate(dataloader):
            X, y, a1_freq = X.to(device), y.to(device), a1_freq.to(device)

            batch_size = X.shape[0]
            encoder_hidden = encoder.init_hidden(batch_size)

            input_length = X.shape[1]
            target_length = y.shape[1]

            loss = 0

            encoder_outputs = torch.zeros(batch_size, input_length, 40, device=device)
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(X[:, ei, :], encoder_hidden)
                encoder_outputs[:, ei] += encoder_output[:, 0]

            decoder_input = torch.zeros((y.shape[0]), dtype=torch.int, device=device)
            decoder_hidden = encoder_hidden

            predicts = []
            label = []
            for di in range(target_length):
                decoder_logit, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion['CrossEntropy'](decoder_logit[0], y[:, di])
                predict = torch.squeeze(torch.argmax(F.softmax(decoder_logit, dim=-1), dim=-1))
                decoder_input = predict
                predicts.append(predict)
            
            test_loss = loss.item()
            predictions.append(torch.stack(predicts).T)
            labels.append(y)
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    n_samples = len(labels)
    _r2_score = sum([matthews_corrcoef(labels[i].cpu().detach().numpy(), predictions[i].cpu().detach().numpy()) for i in range(n_samples)])/n_samples

    return test_loss/target_length , _r2_score