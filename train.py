import os
import json
import torch
from sklearn.metrics import r2_score
from model.custom_cross_entropy import CustomCrossEntropyLoss
from model.single_model import SingleModel
from model.early_stopping import EarlyStopping
from data.dataset import RegionDataset
from torch.utils.data import DataLoader
from utils.argument_parser import get_argument
from utils.plot_chart import draw_chart
from utils.imputation import train, evaluation, get_device, save_model
torch.manual_seed(42)

def run(dataloader, model_config, args, region, epochs=200):
    device = get_device(args.gpu)
    type_model = args.model_type
    lr = args.learning_rate
    gamma = args.gamma if type_model == 'Higher' else -args.gamma
    output_model_dir = args.output_model_dir
    train_loader = dataloader['train']
    val_loader = dataloader['validation']

    #Init Model
    model = SingleModel(model_config, device, type_model=type_model).to(device)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameters:",count_parameters(model))
    
    loss_fn = CustomCrossEntropyLoss(gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    early_stopping = EarlyStopping(patience=10)

    #Start train
    _r2_score_list, loss_values = [], [] #train
    r2_test_list, test_loss_list = [], [] #validation
    best_test_r2 = -99999999
    for t in range(epochs):
        train_loss, r2_train = train(train_loader, model, device, loss_fn, optimizer, scheduler)
        test_loss, r2_test = evaluation(val_loader, model, device, loss_fn)
        loss_values.append(train_loss)
        _r2_score_list.append(r2_train)
        r2_test_list.append(r2_test)
        test_loss_list.append(test_loss)
        print(f"[REGION {region} - EPOCHS {t+1}]: train_loss: {train_loss:>7f}, train_r2: {r2_train:>7f}, test_loss: {test_loss:>7f}, test_r2: {r2_test:>7f}")
        
        # Save best model
        if r2_test > best_test_r2:
            best_test_r2 = r2_test
            best_epochs = t+1
            save_model(model, region, type_model, output_model_dir, best=True)

        #Early stopping
        if args.early_stopping:
            early_stopping(test_loss)
            if early_stopping.early_stop:
                break
    print(f"Best model at epochs {best_epochs} with R2 score: {best_test_r2}")
    draw_chart(loss_values, _r2_score_list, test_loss_list, r2_test_list, region, type_model)
    save_model(model, region, type_model, output_model_dir)

def main():
    args = get_argument()
    root_dir = args.root_dir
    model_config_dir = args.model_config_dir
    batch_size = args.batch_size
    epochs = args.epochs
    chromosome = args.chromosome
    regions = args.regions.split("-")

    with open(os.path.join(root_dir, 'index.txt'),'w+') as index_file:
        index_file.write("0")

    for region in range(int(regions[0]), int(regions[-1])+1):
        print(f"----------Training Region {region}----------")
        with open(os.path.join(model_config_dir, f'region_{region}_config.json'), "r") as json_config:
            model_config = json.load(json_config)
            model_config['region'] = region
        dataset = RegionDataset(root_dir, region, chromosome)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        print("[Train - Test]:", len(train_set), len(val_set))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        dataloader = {'train': train_loader, 'validation': val_loader}
        run(
            dataloader,
            model_config,
            args, 
            region,
            epochs
        )

if __name__ == "__main__":
    main()
