import torch 
from torch import nn
from torch._C import dtype
from .gru_model import GRUModel
from torch.nn import functional as F
from .activations import get_activation

TYPE_MODEL = ['Higher', 'Lower']
class Discriminator(nn.Module):
    def __init__(self, model_config, activation='gelu'):
        super().__init__()
        hidden_size = model_config['num_classes']
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dense_prediction = nn.Linear(hidden_size, 1)
        self.activation = get_activation(activation)
    
    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self.activation(hidden_states)
        logits = torch.squeeze(self.dense_prediction(hidden_states), dim=-1)
        return logits


class SingleModel(nn.Module):
    def __init__(self,model_config, device, type_model=None):
        super(SingleModel,self).__init__()
        assert type_model in TYPE_MODEL
        self.num_classes = model_config['num_classes']
        self.num_outputs = model_config['num_outputs']
        self.type_model = type_model
        self.generator = GRUModel(model_config, device, type_model=self.type_model)
        self.discriminator = Discriminator(model_config, 'sigmoid')


    def forward(self, input_):
        logit_prediction = self.generator(input_)
        logit_generator = torch.cat(logit_prediction, dim=0)
        fake_input = F.softmax(torch.stack(logit_prediction), dim=-1)
        discriminator_logit = torch.round(torch.sign(self.discriminator(torch.stack(logit_prediction))+1)/2)
        return logit_generator, fake_input, discriminator_logit
    



    