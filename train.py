import os
import json
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import r2_score
from model.hybrid_model import HybridModel
from data.dataset import RegionDataset
from torch import nn
from torch.utils.data import DataLoader
torch.manual_seed(42)

def evaluation(dataloader, model, device, loss_fn):
    #TODO
    '''
        Evaluate model with R square score
    '''
    model.eval()
    _r2_score = 0
    test_loss = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            label = torch.reshape(y, (-1,2)).long()

            # Compute prediction error
            logits, prediction = model(X.float())
            loss = loss_fn(logits, label[:,1])
            _r2_score += r2_score(
                torch.argmax(prediction,dim=1).cpu().detach().numpy(),
                label[:,1].cpu().detach().numpy()
            )
            test_loss += loss.item()

    test_loss /= batch+1
    _r2_score /= batch+1
    return test_loss, _r2_score


def draw_chart(train_loss, train_r2_score, val_loss, val_r2_score, region):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f'Region {region}')
    axs[0,0].set_title("Training Loss")
    axs[0,0].plot(train_loss)
    axs[1,0].set_title("Training R2 score")
    axs[1,0].plot(train_r2_score)
    axs[0,1].set_title("Validation Loss")
    axs[0,1].plot(val_loss)
    axs[1,1].set_title("Validation R2 score")
    axs[1,1].plot(val_r2_score)
    fig.tight_layout()
    plt.savefig(f"images/region_{region}.png")

def train(dataloader, a1_freq_list, model_config, args, region, batch_size=1, epochs=200):
    if args.gpu == True:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == 'cpu': 
            print("You don't have GPU to impute genotype, so I will run by CPU")
        else:
            print(f"You're using GPU {torch.cuda.get_device_name(0)} to impute genotype")
    else: 
        device = 'cpu'
        print("You're using CPU to impute genotype")
    model_type = args.model_type
    lr = args.learning_rate
    a1_freq_list = torch.tensor(a1_freq_list.tolist()*batch_size)
    train_loader = dataloader['train']
    val_loader = dataloader['validation']
    #Init Model
    model = HybridModel(model_config, a1_freq_list, batch_size=batch_size, mode=model_type).float().to(device)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameters:",count_parameters(model))
    # loss_fn = model.CustomCrossEntropyLoss
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    _r2_score_list, loss_values = [], [] #train
    r2_test_list, test_loss_list = [], [] #validation
    for t in range(epochs):
        _r2_score = 0
        train_loss = 0
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            label = torch.reshape(y, (-1,2)).long()

            # Compute prediction error
            logits, prediction = model(X.float())
            loss = loss_fn(logits, label[:,1])
            _r2_score += r2_score(
                torch.argmax(prediction,dim=1).cpu().detach().numpy(),
                label[:,1].cpu().detach().numpy()
            )
            train_loss += loss.item()
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        _r2_score /= batch+1
        train_loss /= batch+1
        test_loss, r2_test = evaluation(val_loader, model, device, loss_fn)

        loss_values.append(loss)
        _r2_score_list.append(_r2_score)

        test_loss_list.append(test_loss)
        r2_test_list.append(r2_test)
        print(f"[REGION {region} - EPOCHS {t+1}]: train_loss: {train_loss:>7f}, train_r2: {_r2_score:>7f}, test_loss: {test_loss:>7f}, test_r2: {r2_test:>7f}")

    draw_chart(loss_values, _r2_score_list, test_loss_list, r2_test_list, region)


def main():
    description = 'Genotype Imputation'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--root-dir', type=str, required=True,
                        dest='root_dir', help='Data folder')
    parser.add_argument('--model-config-dir', type=str, required=True,
                        dest='model_config_dir', help='Model config folder')
    parser.add_argument('--model-type', type=str, required=True,
                        dest='model_type', help='Model type')
    parser.add_argument('--gpu', type=bool, default=False, required=False,
                        dest='gpu', help='Using GPU')
    parser.add_argument('--batch-size', type=int, default=2, required=False,
                        dest='batch_size', help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, required=False,
                        dest='epochs', help='Epochs')
    parser.add_argument('--regions', type=str, default=1, required=False,
                        dest='regions', help='Region range')
    parser.add_argument('--chr', type=str, default='chr22', required=False,
                        dest='chromosome', help='Chromosome')
    parser.add_argument('--lr', type=int, default=1e-4, required=False,
                        dest='learning_rate', help='Learning rate')
    args = parser.parse_args()

    root_dir = args.root_dir
    model_config_dir = args.model_config_dir
    batch_size = args.batch_size
    epochs = args.epochs
    chromosome = args.chromosome
    regions = args.regions.split("-")

    for region in range(int(regions[0]), int(regions[-1])+1):
        print(f"----------Training Region {region}----------")
        with open(os.path.join(model_config_dir, f'region_{region}_config.json'), "r") as json_config:
            model_config = json.load(json_config)
        dataset = RegionDataset(root_dir, region, chromosome)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        print("[Train - Test]:", len(train_set), len(val_set))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        dataloader = {'train': train_loader, 'validation': val_loader}
        train(
            dataloader,
            dataset.a1_freq_list,
            model_config,
            args, region,
            batch_size,
            epochs
        )

if __name__ == "__main__":
    main()
