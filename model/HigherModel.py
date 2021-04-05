import numpy as np
import torch
from torch import nn

class HigherModel(nn.Module):
    def __init__(self, model_config, bidir=True):
        super(HigherModel,self).__init__()
        self.input_dim = model_config['input_dim']
        self.hidden_units = model_config['num_units']
        self.num_classes = model_config['num_classes']
        self.num_outputs = model_config['num_outputs']
        self.num_layers = model_config['num_layers']
        self.feature_size = model_config['feature_size']
        self.num_inputs = model_config['num_inputs']
        self.output_points_fw = model_config['output_points_fw']
        self.output_points_bw = model_config['output_points_bw']
        self.bidir = bidir

        self.gru = nn.GRU(
            input_size = self.feature_size,
            hidden_size = self.hidden_units,
            num_layers = self.num_layers,
            bidirectional = self.bidir)

    def forward(self, input_, hidden=None):
        '''
            return logits_list(g)  in paper
        '''
        gru_inputs = []
        outputs_fw = [None] * self.num_inputs
        outputs_bw = [None] * self.num_inputs
        features = torch.zeros((self.num_inputs, 2, self.feature_size))

        for i in range(self.num_inputs):
            gru_input = torch.matmul(input_[0][i], features[i])
            gru_inputs.append(gru_input)
        input_ = torch.reshape(torch.stack(gru_inputs),(1, -1,self.feature_size))
        logits, state = self.gru(input_, hidden)

        outputs_fw = logits[0,:, :self.hidden_units]
        outputs_bw = logits[0,:, self.hidden_units:]

        logit_list = []
        for i, (t_fw, t_bw) in enumerate(zip(self.output_points_fw, self.output_points_bw)):
            gru_output = []
            if t_fw is not None:
                gru_output.append(outputs_fw[t_fw])
            if t_bw is not None:
                gru_output.append(outputs_bw[t_bw])
            if len(gru_output) > 1:
                gru_output = torch.reshape(torch.stack(gru_output),(1,-1))
            else: 
                gru_output = torch.stack(gru_output)

            num_classes = self.num_classes
            a = torch.rand((gru_output.shape[1], num_classes), requires_grad=True)
            b = torch.rand((num_classes), requires_grad=True)
            logit = torch.matmul(gru_output, a) + b
            logit_list.append(logit)

        return torch.stack(logit_list), state

    def init_hidden(self, number_of_variants):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers*2 , number_of_variants, self.hidden_units).zero_() # self.num_layers*2(bidirection)
        return hidden
