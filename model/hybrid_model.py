from .higher_model import HigherModel
from .lower_model import LowerModel
from .custom_cross_entropy import CustomCrossEntropyLoss
import torch 
from torch import nn 

MODEL_NAME = ['Hybrid', 'Higher', 'Lower']
class HybridModel(nn.Module):
    def __init__(self,model_config, a1_freq_list, batch_size=1, bidir=True, mode=None):
        super(HybridModel,self).__init__()
        assert mode in MODEL_NAME
        self.input_size = model_config['input_dim']
        self.hidden_units = model_config['num_units']
        self.num_classes = model_config['num_classes']
        self.num_outputs = model_config['num_outputs']
        self.mode = mode
        self.bidir = bidir
        self.batch_size = batch_size

        if self.mode == 'Higher':
            gramma = 1
            self.higherModel = HigherModel(model_config, batch_size=self.batch_size, bidir=self.bidir)
        elif self.mode == 'Lower':
            gramma = -0.05
            self.lowerModel = LowerModel(model_config, bidir=self.bidir)
        else:
            gramma = 0
            self.higherModel = HigherModel(model_config, bidir=self.bidir)
            self.lowerModel = LowerModel(model_config, bidir=self.bidir)

        self.CustomCrossEntropyLoss = CustomCrossEntropyLoss(a1_freq_list, gramma=gramma)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_):
        # print("Hii", input_.shape)
        number_of_variants = input_.shape[1]
        num_classes = self.num_classes

        #Higher Model
        if self.mode == 'Higher':
            init_hidden_higher = self.higherModel.init_hidden(number_of_variants)
            logits, _ = self.higherModel(input_, init_hidden_higher.data)
            logit = torch.reshape(logits, shape=[-1, self.num_outputs, num_classes]) #pre-softmax
            pred = torch.reshape(self.softmax(logits), shape=[-1, self.num_outputs, num_classes]) #softmax
            return logit, pred

        #Lower Model
        elif self.mode == 'Lower':
            init_hidden_lower = self.lowerModel.init_hidden(number_of_variants)
            logits, _ = self.lowerModel(input_, init_hidden_lower.data)
            return torch.reshape(
                logits,
                shape=[-1, self.num_outputs, num_classes])

        #Hybrid Model
        else:
            #Higher Model
            init_hidden_higher = self.higherModel.init_hidden(number_of_variants)
            logits_higher, _ = self.higherModel(input_, init_hidden_higher.data)

            #Lower Model
            init_hidden_lower = self.lowerModel.init_hidden(number_of_variants)
            logits_lower, _ = self.lowerModel(input_, init_hidden_lower.data)

            #concatenate 2 Lower and Higher models
            logits = torch.cat((logits_lower, logits_higher), dim=2)

            logit_list = [] 
            for input_ in logits:
                c = torch.randn((num_classes*2, num_classes), requires_grad=True)
                d = torch.randn((num_classes), requires_grad=True)
                logit = torch.matmul(input_, c) + d
                logit_list.append(logit)
            predictions = torch.reshape(
                    torch.stack(logit_list),
                    shape=[-1, self.num_outputs, num_classes])
            return predictions