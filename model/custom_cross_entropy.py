import torch 
from torch import nn
from torch import autograd

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, a1_freq_list, gramma=0):
        super(CustomCrossEntropyLoss, self).__init__()
        self.gramma = gramma
        self.a1_freq_list = a1_freq_list

    def forward(self, x, y0):
        loss = 0.
        n_batch, n_class = x.shape
        # print(n_class)
        for x1,y1, a1_freq in zip(x,y0, self.a1_freq_list):
            class_index = int(y1.item())
            loss = loss + ((2*a1_freq)**(self.gramma))*torch.log(torch.exp(x1[class_index])/(torch.exp(x1).sum()))
        loss = - loss/n_batch
        return loss
