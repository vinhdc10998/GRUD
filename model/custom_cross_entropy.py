import torch 
from torch import nn

class CustomCrossEntropy(nn.Module):
    def __init__(self, gramma=0):
        super(CustomCrossEntropy, self).__init__()
        self.gramma = gramma

    def forward(self, predictions, labels, a1_freq):
        print("[DEBUG LOSS FUNCTION]:", predictions.shape, labels.shape)
        print(-1*torch.sum(labels*torch.log(predictions).T)/len(labels))
        return -1*torch.sum(labels*torch.log(predictions.T))/len(labels)