import torch 
from torch import nn

class CustomCrossEntropy(nn.Module):
    def __init__(self, gramma=0):
        super(CustomCrossEntropy, self).__init__()
        self.gramma = gramma

    def forward(self, predictions, labels, a1_freq):
        print("[DEBUG LOSS FUNCTION]:", predictions.shape, labels.shape)
        print(-1*torch.sum(labels*torch.log(predictions.T))/len(labels))
        # pause = input("PAUSE......")
        # tmp = -1*torch.sum(labels * torch.log(predictions), dim=-1).mean()
        # print("[DEBUS]:", tmp)
        # loss = (-1 * torch.sum()).mean()
        # print(loss)
        return -1*torch.sum(labels*torch.log(predictions.T))/len(labels)