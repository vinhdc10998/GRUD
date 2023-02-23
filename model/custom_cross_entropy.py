from torch import nn
from torch.nn import functional as F

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, gamma=0):
        super(CustomCrossEntropyLoss, self).__init__()
        self.gamma = gamma
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target, a1_freq_list):
        assert pred.shape[0] == target.shape[0] and target.shape[0] == a1_freq_list.shape[0]
        return self.cross_entropy_loss(pred, target).mean()

