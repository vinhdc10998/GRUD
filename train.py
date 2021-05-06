import os
import json
import torch
from model.custom_cross_entropy import CustomCrossEntropyLoss
from model.single_model import SingleModel
from model.early_stopping import EarlyStopping
from data.dataset import RegionDataset
from torch.utils.data import DataLoader
from utils.argument_parser import get_argument
from utils.plot_chart import draw_chart
from utils.imputation import train, evaluation, get_device, save_model
torch.manual_seed(42)

def run(dataloader, model_config, args, region):
    device = get_device(args.gpu)
    type_model = args.model_type
    lr = args.learning_rate
    epochs = args.epochs
    gamma = args.gamma if type_model == 'Higher' else -args.gamma
    output_model_dir = args.output_model_dir
    train_loader = dataloader['train']
    val_loader = dataloader['val']
    test_loader = dataloader['test']

    #Init Model
    model = SingleModel(model_config, device, type_model=type_model).to(device)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameters:",count_parameters(model))
    loss_fn = CustomCrossEntropyLoss(gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    # early_stopping = EarlyStopping(patience=10)

    #Start train
    _r2_score_list, loss_values = [], [] #train
    r2_val_list, val_loss_list = [], [] #validation
    best_val_loss = 99999999 
    for t in range(epochs):
        train_loss, r2_train = train(train_loader, model, device, loss_fn, optimizer, scheduler)
        val_loss, r2_val, _ = evaluation(val_loader, model, device, loss_fn)
        test_loss, r2_test, _ = evaluation(test_loader, model, device, loss_fn)
        loss_values.append(train_loss)
        _r2_score_list.append(r2_train)
        r2_val_list.append(r2_val)
        val_loss_list.append(val_loss)
        print(f"[REGION {region} - EPOCHS {t+1}]\
            lr: {optimizer.param_groups[0]['lr']}\
                train_loss: {train_loss:>7f}, train_r2: {r2_train:>7f},\
                    val_loss: {val_loss:>7f}, val_r2: {r2_val:>7f},\
                        test_loss: {test_loss:>7f}, test_r2: {r2_test:>7f}")   
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epochs = t+1
            save_model(model, region, type_model, output_model_dir, best=True)

        #Early stopping
        # if args.early_stopping:
        #     early_stopping(val_loss)
        #     if early_stopping.early_stop:
        #         break

    print(f"Best model at epochs {best_epochs} with loss: {best_val_loss}")
    draw_chart(loss_values, _r2_score_list, val_loss_list, r2_val_list, region, type_model)
    save_model(model, region, type_model, output_model_dir)

def main():
    args = get_argument()
    root_dir = args.root_dir
    model_config_dir = args.model_config_dir
    batch_size = args.batch_size
    epochs = args.epochs
    chromosome = args.chromosome
    regions = args.regions.split("-")
    test_dir = args.test_dir

    with open(os.path.join(root_dir, 'index.txt'), 'w+') as index_file:
        index_file.write("0")
    with open(os.path.join(test_dir, 'index.txt'), 'w+') as index_file:
        index_file.write("0")

    for region in range(int(regions[0]), int(regions[-1])+1):
        print(f"----------Training Region {region}----------")
        with open(os.path.join(model_config_dir, f'region_{region}_config.json'), "r") as json_config:
            model_config = json.load(json_config)
            model_config['region'] = region
        train_val_set = RegionDataset(root_dir, region, chromosome)
        test_set = RegionDataset(test_dir, region, chromosome)
        train_size = int(0.8 * len(train_val_set))
        val_size = len(train_val_set) - train_size
        train_set, val_set = torch.utils.data.random_split(train_val_set, [train_size, val_size])

        print("[Train - Val- Test]:", len(train_set), len(val_set), len(test_set), 'samples')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        dataloader = {
            'train': train_loader, 
            'test': test_loader,
            'val': val_loader
        }
        run(
            dataloader,
            model_config,
            args, 
            region
        )

if __name__ == "__main__":
    main()
