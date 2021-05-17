import torch 
from torch import nn
from .gru_model import GRUModel
from torch.nn import functional as F

TYPE_MODEL = ['Hybrid']
class MultiModel(nn.Module):
    def __init__(self,model_config, device, type_model=None):
        super(MultiModel,self).__init__()
        assert type_model in TYPE_MODEL
        self.num_classes = model_config['num_classes']
        self.num_outputs = model_config['num_outputs']
        self.type_model = type_model

        self.lowerModel = GRUModel(model_config, device, type_model='Lower')
        self.higherModel = GRUModel(model_config, device, type_model='Higher')

        if self.train():
            self.lowerModel.load_state_dict(self.get_gru_layer(model_config['lower_path'], device))
            self.higherModel.load_state_dict(self.get_gru_layer(model_config['higher_path'], device))

            for param in self.lowerModel.parameters():
                param.requires_grad = False
            for param in self.higherModel.parameters():
                param.requires_grad = False
    
        self.linear = nn.ModuleList([nn.Linear(self.num_classes*2, self.num_classes) for _ in range(self.num_outputs)])


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
        logits_1 = self.higherModel(input_)
        logits_2 = self.lowerModel(input_)
        logits = torch.cat((torch.stack(logits_1), torch.stack(logits_2)), dim=-1)
        logit_list = []
        for index, logit in enumerate(logits):
            logit_tmp = self.linear[index](logit)
            logit_list.append(logit_tmp)
                
        logit = torch.cat(logit_list, dim=0)
        pred = F.softmax(torch.stack(logit_list), dim=-1)
        return logit, pred

    
    