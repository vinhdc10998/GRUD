import torch 
from torch import nn
from torch import nn
from torch.nn import functional as F

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, a1_freq_list, device, gramma=0):
        super(CustomCrossEntropyLoss, self).__init__()
        self.gramma = gramma
        self.a1_freq_list = a1_freq_list
        self.device = device

    def forward(self, pred, target):
        batch_size = len(target) // len(self.a1_freq_list)
        a1_freq_list = torch.tensor(self.a1_freq_list.tolist()*batch_size)
        maf = ((2*a1_freq_list)**self.gramma).to(self.device)
        y_true = (maf * target).type(torch.LongTensor)
        loss = nn.NLLLoss().to(self.device)
        return loss(F.log_softmax(pred, dim=1) , y_true)
