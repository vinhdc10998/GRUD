import torch 
from torch import nn
from .gru_model import GRUModel
from torch.nn import functional as F
from activations import get_activation

TYPE_MODEL = ['Higher', 'Lower']
class ElectraDiscriminator(nn.Module):
    def __init__(self, model_config, activation='gelu'):
        super().__init__()
        hidden_size = model_config['num_outputs']
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dense_prediction = nn.Linear(hidden_size, 1)
        self.activation = get_activation(activation)
    
    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self.activation(hidden_states)
        logits = torch.squeeze(self.dense_prediction(hidden_states), -1)
        return logits


class SingleModel(nn.Module):
    def __init__(self,model_config, device, type_model=None):
        super(SingleModel,self).__init__()
        assert type_model in TYPE_MODEL
        self.num_classes = model_config['num_classes']
        self.num_outputs = model_config['num_outputs']
        self.type_model = type_model
        self.discriminator = ElectraDiscriminator(model_config, 'gelu')

        self.gruModel = GRUModel(model_config, device, type_model=self.type_model)

    def forward(self, input_):
        logit_list = self.gruModel(input_)
        logit = torch.cat(logit_list, dim=0)
        prediction = F.softmax(torch.stack(logit_list), dim=-1)
        return logit, prediction
    
    