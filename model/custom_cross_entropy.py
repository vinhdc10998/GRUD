from torch import nn
from torch.nn import functional as F

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, gamma=0):
        super(CustomCrossEntropyLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred, target, a1_freq_list):
        #TODO:
        # assert pred, target, a1_freq_list
        loss = nn.CrossEntropyLoss(reduction='none')
        return (((2*a1_freq_list)**self.gamma) * loss(pred, target)).mean()
