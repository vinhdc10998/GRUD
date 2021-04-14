import torch 
from torch import nn
from torch import nn
from torch.nn import functional as F
from torch import autograd

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, a1_freq_list, gramma=0):
        super(CustomCrossEntropyLoss, self).__init__()
        self.gramma = gramma
        self.a1_freq_list = a1_freq_list

    def forward(self, pred, target):
        batch_size = len(target) // len(self.a1_freq_list)
        a1_freq_list = torch.tensor(self.a1_freq_list.tolist()*batch_size)
        target = (((2*a1_freq_list)**self.gramma) * target).long() 

        loss = nn.NLLLoss()
        return loss(F.log_softmax(pred, dim=1), target)
