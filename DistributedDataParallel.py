import os
import torch
import torch.distributed as dist
from torch.nn import MSELoss
from torch.optim import Adam
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn

def get_dataset(mode):
    if mode=='train':
        data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )
    else:
        data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )

    return data

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def prepare(rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    dataset = get_dataset("train")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    return dataloader

def cleanup():
    dist.destroy_process_group()


def test(rank, world_size):
    # setup the process groups
    torch.cuda.set_device(rank)

    print(rank, world_size)
    setup(rank, world_size)
    # prepare the dataloader
    dataloader = prepare(rank, world_size)
    # instantiate the model(it's your own model) and move it to the right device
    model = NeuralNetwork().to(rank)
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(100):
        
        # if we are using DistributedSampler, we have to tell it which epoch this is
        dataloader.sampler.set_epoch(epoch)       
        
        for step, (x,y) in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            x, y = x.to(rank), y.to(rank)
            pred = model(x)
            label = y
            
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
        print(f"EPOCH: {epoch}\t LOSS: {loss}")
    cleanup()

if __name__ == '__main__':
    # suppose we have 3 gpus
    world_size = 1
    mp.spawn(
        test,
        args=[world_size],
        nprocs=world_size
    )
