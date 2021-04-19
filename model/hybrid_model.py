from .gru_model import GRUModel
from .custom_cross_entropy import CustomCrossEntropyLoss
import torch 
from torch import nn 

TYPE_MODEL = ['Hybrid', 'Higher', 'Lower']
class HybridModel(nn.Module):
    def __init__(self,model_config, a1_freq_list, device, type_model=None):
        super(HybridModel,self).__init__()
        assert type_model in TYPE_MODEL
        self.num_classes = model_config['num_classes']
        self.num_outputs = model_config['num_outputs']
        self.type_model = type_model

        gammar = 0
        if self.type_model == 'Higher' or self.type_model == 'Lower':
            gammar = 0.01
            gammar = gammar if self.type_model == 'Higher' else -gammar
            self.gruModel = GRUModel(model_config, device, type_model=self.type_model)
        else:
            #TODO
            #Train higher and lower model to get weight models
            assert False
            self.higherModel = GRUModel(model_config, device, type_model='Higher')
            self.lowerModel = GRUModel(model_config, device, type_model='Lower')

        self.CustomCrossEntropyLoss = CustomCrossEntropyLoss(a1_freq_list, device, gammar)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_):
        batch = input_.shape[0]
        #Higher or Lower Model
        if self.type_model == 'Higher' or self.type_model == 'Lower':
            init_hidden = self.gruModel.init_hidden(batch)
            logits = self.gruModel(input_, init_hidden)
            logit = torch.cat(logits, dim=0)
            pred = self.softmax(torch.stack(logits))
            return logit, pred
        return None
