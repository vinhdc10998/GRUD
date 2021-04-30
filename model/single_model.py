import torch 
from torch import nn
from .gru_model import GRUModel
from torch.nn import functional as F

TYPE_MODEL = ['Higher', 'Lower']
class SingleModel(nn.Module):
    def __init__(self,model_config, device, type_model=None):
        super(SingleModel,self).__init__()
        assert type_model in TYPE_MODEL
        self.num_classes = model_config['num_classes']
        self.num_outputs = model_config['num_outputs']
        self.type_model = type_model

        self.gruModel = GRUModel(model_config, device, type_model=self.type_model)

    def forward(self, input_):
        logit_list = self.gruModel(input_)
        logit = torch.reshape(logit_list, shape=[-1, self.num_classes])
        prediction = F.softmax(logit_list, dim=-1)
        return logit, prediction
    
    