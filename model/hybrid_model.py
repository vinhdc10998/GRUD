import torch 
from torch import nn 
from .higher_model import HigherModel
from .lower_model import LowerModel
from .custom_cross_entropy import CustomCrossEntropy

MODEL_NAME = ['Hybrid', 'Higher', 'Lower']
class HybridModel(nn.Module):
    def __init__(self,model_config, batch_size=1, bidir=True, mode=None):
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
            self.CrossEntropy = CustomCrossEntropy(0.5)
        elif self.mode == 'Lower':
            self.CrossEntropy = CustomCrossEntropy(-0.5)
        else:
            self.CrossEntropy = CustomCrossEntropy()

        self.higherModel = HigherModel(model_config, bidir=self.bidir)
        self.lowerModel = LowerModel(model_config, bidir=self.bidir)
        self.softmax = nn.Softmax()

    def forward(self, input_):
        number_of_variants = input_.size()[1]
        num_classes = self.num_classes

        if self.mode == 'Hybrid':
            #Higher Model
            init_hidden_higher = self.higherModel.init_hidden(number_of_variants)
            logits_higher, _ = self.higherModel(input_, init_hidden_higher.data)

            #Lower Model
            init_hidden_lower = self.lowerModel.init_hidden(number_of_variants)
            logits_lower, _ = self.lowerModel(input_, init_hidden_lower.data)

            #concatenate 2 Lower and Higher models
            logits = torch.cat((logits_lower, logits_higher), dim=2)
            dim_logits = num_classes*2

        elif self.mode == 'Higher':
            #Higher Model
            init_hidden_higher = self.higherModel.init_hidden(number_of_variants)
            logits, _ = self.higherModel(input_, init_hidden_higher.data)
            dim_logits = num_classes

        elif self.mode == 'Lower':
            #Lower Model
            init_hidden_lower = self.lowerModel.init_hidden(number_of_variants)
            logits, _ = self.lowerModel(input_, init_hidden_lower.data)
            dim_logits = num_classes

        logit_list = [] 
        for input_ in logits:
            c = torch.rand((dim_logits, num_classes), requires_grad=True)
            d = torch.rand((num_classes), requires_grad=True)
            logit = torch.matmul(input_, c) + d
            logit_list.append(logit)

        predictions = torch.reshape(
                self.softmax(torch.stack(logit_list)),
                shape=[-1, self.num_outputs, num_classes])

        return predictions