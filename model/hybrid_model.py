from .gru_model import GRUModel
from .custom_cross_entropy import CustomCrossEntropyLoss
import torch 
from torch import nn 

TYPE_MODEL = ['Hybrid', 'Higher', 'Lower']
class HybridModel(nn.Module):
    def __init__(self,model_config, a1_freq_list, batch_size=1, bidir=True, type_model=None):
        super(HybridModel,self).__init__()
        assert type_model in TYPE_MODEL
        self.input_size = model_config['input_dim']
        self.hidden_units = model_config['num_units']
        self.num_classes = model_config['num_classes']
        self.num_outputs = model_config['num_outputs']
        self.type_model = type_model
        self.bidir = bidir
        self.batch_size = batch_size

        if self.type_model == 'Higher' or self.type_model == 'Lower':
            self.gruModel = GRUModel(model_config, type_model=self.type_model)
        else:
            self.higherModel = GRUModel(model_config, type_model='Higher')
            self.lowerModel = GRUModel(model_config, type_model='Lower')

        self.CustomCrossEntropyLoss = CustomCrossEntropyLoss(a1_freq_list)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_):
        batch = input_.shape[0]

        #Higher Model
        if self.type_model == 'Higher' or self.type_model == 'Lower':
            init_hidden_higher = self.gruModel.init_hidden(batch)
            logits = self.gruModel(input_, init_hidden_higher) #raw output
            logit = torch.cat(logits, dim=0)
            pred = torch.reshape(self.softmax(logit), shape=[self.num_outputs, -1, self.num_classes]) 
            return logit, pred
        
        return None
