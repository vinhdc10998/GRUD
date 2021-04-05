import numpy as np
import torch 
from torch import nn 
from HigherModel import HigherModel
from LowerModel import LowerModel

class HybridModel(nn.Module):
    def __init__(self,model_config, batch_size=1, bidir=True):
        super(HybridModel,self).__init__()
        self.input_size = model_config['input_dim']
        self.hidden_units = model_config['num_units']
        self.num_classes = model_config['num_classes']
        self.num_outputs = model_config['num_outputs']

        self.bidir = bidir
        self.batch_size = batch_size

        self.higherModel = HigherModel(model_config, bidir=self.bidir)
        self.lowerModel = LowerModel(model_config, bidir=self.bidir)
        self.softmax = nn.Softmax()

    def forward(self, input_):
        number_of_variants = input_.size()[1]
        
        #Higher Model
        init_hidden_higher = self.higherModel.init_hidden(number_of_variants)
        logits_higher, _ = self.higherModel(input_, init_hidden_higher.data)

        #Lower Model
        init_hidden_lower = self.lowerModel.init_hidden(number_of_variants)
        logits_lower, _ = self.lowerModel(input_, init_hidden_lower.data)

        #concatenate 2 Lower and Higher models
        logits = torch.cat((logits_lower, logits_higher), dim=2)

        logit_list = []
        num_classes = self.num_classes
        for i, input_ in enumerate(logits):
            c = torch.rand((num_classes*2, num_classes), requires_grad=True)
            d = torch.rand((num_classes), requires_grad=True)
            logit = torch.matmul(input_, c) + d
            logit_list.append(logit)

        predictions = torch.reshape(
                self.softmax(torch.stack(logit_list)),
                shape=[-1, self.num_outputs, num_classes])

        return predictions


