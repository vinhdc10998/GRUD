import torch 
from torch import nn
from torch._C import dtype
from .gru_model import GRUModel
from torch.nn import functional as F
from .activations import get_activation

class Discriminator(nn.Module):
    def __init__(self, hidden_size, activation='gelu'):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dense_prediction = nn.Linear(hidden_size, 1)
        self.activation = get_activation(activation)
    
    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self.activation(hidden_states)
        logits = torch.squeeze(self.dense_prediction(hidden_states), dim=-1)
        return logits


class GRUD(nn.Module):
    def __init__(self,model_config, device):
        super(GRUD,self).__init__()
        self.num_classes = model_config['num_classes']
        self.generator = GRUModel(model_config, device)
        self.discriminator = Discriminator(self.num_classes, 'sigmoid')


    def forward(self, input_):
        discriminator_logit = None
        logit_prediction = self.generator(input_)
        fake_logit = torch.stack(logit_prediction)
        logit_generator = torch.cat(logit_prediction, dim=0)
        prediction = F.softmax(fake_logit, dim=-1)
        if self.training:
            discriminator_logit = self.discriminator(fake_logit.detach()).T
        return logit_generator, prediction, discriminator_logit
    



    