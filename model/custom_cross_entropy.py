import torch 
from torch import nn
from torch import nn
from torch._C import device
from torch.nn import functional as F
from torch import autograd

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, a1_freq_list, device, gramma=0):
        super(CustomCrossEntropyLoss, self).__init__()
        self.gramma = gramma
        self.a1_freq_list = a1_freq_list
        self.device = device

    def forward(self, pred, target):
        batch_size = len(target) // len(self.a1_freq_list)
        a1_freq_list = torch.tensor(self.a1_freq_list.tolist()*batch_size)
        x = ((2*a1_freq_list)**self.gramma).to(self.device)
        target = (x * target).long() 
        loss = nn.NLLLoss().to(self.device)
        return loss(F.log_softmax(pred, dim=1), target)
