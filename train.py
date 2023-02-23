import os
import json
import torch
import numpy as np
import torch.distributed as dist
from torch import nn
from model.custom_cross_entropy import CustomCrossEntropyLoss
from model.grud_model import GRUD
from model.early_stopping import EarlyStopping
from data.dataset import RegionDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

from utils.argument_parser import get_argument
from utils.plot_chart import draw_chart
from utils.imputation import save_check_point, train, evaluation, get_device, save_model
torch.manual_seed(42)
SINGLE_MODEL = ['Higher', 'Lower']
MULTI_MODEL = ['Hybrid']

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


def run(dataloader, model_config, args, region):
    device = get_device(args.gpu)
    lr = args.learning_rate
    epochs = args.epochs

    output_model_dir = args.output_model_dir
    train_loader = dataloader['train']
    val_loader = dataloader['val']
    test_loader = dataloader['test']

    #Init Model
    # if type_model in ['Lower', 'Higher']:
    #     gamma = args.gamma if type_model == 'Higher' else -args.gamma

    model = GRUD(model_config, device).to(device) 
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameters:",count_parameters(model))
    loss_fn = CustomCrossEntropyLoss()
    loss_fct = nn.BCEWithLogitsLoss()
    loss = {
        'CustomCrossEntropy': loss_fn, 
        'BCEWithLogitsLoss': loss_fct
    }
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    early_stopping = EarlyStopping(patience=30)
    check_point_dir = args.check_point_dir
    _r2_score_list, loss_values = [], [] #train
    r2_val_list, val_loss_list = [], [] #validation
    test_loss_list, test_r2_list = [], []
    best_val_loss = np.inf
    start_epochs = 1
    if args.resume and os.path.exists(check_point_dir):
        filename = os.path.join(check_point_dir, os.listdir(check_point_dir)[-1])
        checkpoint = torch.load(filename, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epochs = checkpoint['epoch']+1
    
    # Start training

    print("{:<8}\t{:<15}\t{:>7}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}".format("Region","Epoch", "Learning rate", "Train loss", "Train R2", "Val loss", "Val R2", "Test loss", "Test R2"))
    print("{:<8}\t{:<15}\t{:>7}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}".format("------","-----", "-------------", "----------", "--------", "--------", "------", "---------", "-------"))

    for epoch in range(start_epochs, epochs+1):
        # train_loader.sampler.set_epoch(epoch)       
        # val_loader.sampler.set_epoch(epoch)
        train_loss, r2_train = train(train_loader, model, device, loss, optimizer, scheduler)
        val_loss, r2_val, _ = evaluation(val_loader, model, device, loss)
        # test_loss, r2_test, _ = evaluation(test_loader, model, device, loss)
        test_loss, r2_test = 0, 0
        loss_values.append(train_loss)
        _r2_score_list.append(r2_train)
        r2_val_list.append(r2_val)
        val_loss_list.append(val_loss)
        test_loss_list.append(test_loss)
        test_r2_list.append(r2_test)
        test_loss, r2_test = round(test_loss, 5), round(r2_test, 5)
        val_loss, r2_val = round(val_loss, 5), round(r2_val, 5)
        train_loss, r2_train = round(train_loss, 5), round(r2_train, 5)
        print("{:<8}\t{:<15}\t{:>7f}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}".format(region, epoch, optimizer.param_groups[0]['lr'], train_loss, r2_train, val_loss, r2_val, test_loss, r2_test))
        draw_chart(loss_values, _r2_score_list, val_loss_list, r2_val_list, test_loss_list, test_r2_list, region, output_prefix='Lower_Higher_images')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_model(model, region, output_model_dir, best=True)

        if epoch % 10 == 0 and epoch > 0:
            save_check_point(model, optimizer, epoch, region, check_point_dir)

        # Early stopping
        if args.early_stopping:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                if optimizer.param_groups[0]['lr'] < 1e-6:
                    break
                optimizer.param_groups[0]['lr'] *= 0.5
                early_stopping = EarlyStopping(patience=30)

    print(f"Best model at epoch {best_epoch} with loss: {best_val_loss}")
    save_model(model, region, output_model_dir)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def prepare(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    return dataloader


def main():
    args = get_argument()
    root_dir = args.root_dir
    model_config_dir = args.model_config_dir
    batch_size = args.batch_size
    chromosome = args.chromosome
    regions = args.regions.split("-")
    test_dir = args.test_dir
    # setup(rank, world_size)

    index_region = args.regions + "_GRUD"
    with open(os.path.join(root_dir, f'{index_region}.txt'), 'w+') as index_file:
        index_file.write("0")

    for region in range(int(regions[0]), int(regions[-1])+1):
        print(f"----------Training Region {region}----------")
        with open(os.path.join(model_config_dir, f'region_{region}_config.json'), "r") as json_config:
            model_config = json.load(json_config)
            model_config['region'] = region

        train_val_set = RegionDataset(root_dir, index_region, region, chromosome, dataset=args.dataset)
        # test_set = RegionDataset(test_dir, region, chromosome, dataset=args.dataset)
        train_size = int(0.8 * len(train_val_set))
        val_size = len(train_val_set) - train_size
        train_set, val_set = torch.utils.data.random_split(train_val_set, [train_size, val_size])

        print("[Train - Val- Test]:", len(train_set), len(val_set), 'samples')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)
        # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        # train_loader = prepare(train_set, rank, world_size)
        # val_loader = prepare(val_set, rank, world_size)

        dataloader = {
            'train': train_loader, 
            'test': None,
            'val': val_loader
        }
        run(dataloader, model_config, args, region)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
