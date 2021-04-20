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
            # assert False
            self.higherModel = GRUModel(model_config, device, type_model='Higher')
            self.lowerModel = GRUModel(model_config, device, type_model='Lower')
            self.lowerModel.load_state_dict(self.get_gru_layer(model_config['lower_path'], device))
            self.higherModel.load_state_dict(self.get_gru_layer(model_config['higher_path'], device))
            for param in self.lowerModel.parameters():
                param.requires_grad = False
            for param in self.higherModel.parameters():
                param.requires_grad = False
            self.linear = nn.ModuleList([nn.Linear(self.num_classes*2, self.num_classes) for _ in range(self.num_outputs)])
        self.CustomCrossEntropyLoss = CustomCrossEntropyLoss(a1_freq_list, device, gammar)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def get_gru_layer(path, device):
        tmp = torch.load(path, map_location=torch.device(device))
        a = {}
        for i in tmp:
            if 'gru' in i:
                k = i[9:]
                a[k] = tmp[i]
        return a

    def forward(self, input_):
        batch = input_.shape[0]
        #Higher or Lower Model
        if self.type_model == 'Higher' or self.type_model == 'Lower':
            init_hidden = self.gruModel.init_hidden(batch)
            logit_list = self.gruModel(input_, init_hidden)
        else:
            init_hidden_higher = self.higherModel.init_hidden(batch)
            init_hidden_lower = self.lowerModel.init_hidden(batch)
            logits_1 = self.higherModel(input_, init_hidden_higher)
            logits_2 = self.lowerModel(input_, init_hidden_lower)
            logits = torch.cat((torch.stack(logits_1), torch.stack(logits_2)), dim=-1)
            logit_list = []
            for index, logit in enumerate(logits):
                tmp = self.linear[index](logit)
                logit_list.append(tmp)
        logit = torch.cat(logit_list, dim=0)
        pred = self.softmax(torch.stack(logit_list))
        return logit, pred

    