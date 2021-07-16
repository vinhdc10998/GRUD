from typing import Sequence
import torch
from torch import nn
import math
import time
from torch.nn import functional as F

class GRUModel(nn.Module):
    def __init__(self, model_config, device, type_model):
        super(GRUModel,self).__init__()
        # Debug
        model_config['feature_size'] = 10
        #Debug
        self.input_dim = model_config['input_dim']
        self.hidden_units = model_config['num_units']
        self.num_classes = model_config['num_classes']
        self.num_outputs = model_config['num_outputs']
        # self.feature_size = model_config['feature_size']
        self.feature_size = 20

        self.num_inputs = model_config['num_inputs']
        self.output_points_fw = model_config['output_points_fw']
        self.output_points_bw = model_config['output_points_bw']
        self.region = model_config['region']
        # self.num_layers = model_config['num_layers']
        self.num_layers = 8

        self.num_encode = model_config['num_encode']
        self.device = device
        self.linear_feature_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, self.feature_size, bias=True),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(self.feature_size, self.feature_size*2, bias=True)
            ) for _ in range(self.num_inputs)])
        # self.linear = nn.Linear(self.input_dim, self.feature_size)
        self.batch_norm = nn.BatchNorm1d(self.feature_size*2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        # self.sigmoid = nn.Sigmoid()
        self.batch_norm_list = nn.ModuleList([nn.BatchNorm1d(self.hidden_units*2) for _ in range(self.num_layers)])
        
        self.gru = nn.ModuleList(self._create_gru_cell(
            self.feature_size*2, 
            self.hidden_units,
            self.num_layers
        ))

        self.list_linear = nn.ModuleList(self._create_linear_list(
            self.hidden_units,
            self.num_classes,
            self.output_points_fw,
            self.output_points_bw
        ))
        
        # self.linear_predict =nn.Linear(self.num_encode, self.num_classes)
 
    @staticmethod
    def _create_gru_cell(input_size, hidden_units, num_layers):
        gru = [nn.GRU(input_size, hidden_units, bidirectional=True)] # First layer
        gru += [nn.GRU(hidden_units*2, hidden_units, bidirectional=True) for _ in range(num_layers-1)] # 2 -> num_layers
        return gru

    @staticmethod
    def _create_linear_list(hidden_units, num_classes, output_points_fw, output_points_bw):
        list_linear = []
        for (t_fw, t_bw) in (zip(output_points_fw, output_points_bw)):
            if (t_fw is not None) and (t_bw is not None) and (not math.isnan(t_fw)) and (not math.isnan(t_bw)):
                list_linear.append(nn.Linear(hidden_units*2, num_classes, bias=True)) 
            else:
                list_linear.append(nn.Linear(hidden_units, num_classes, bias=True))
        return list_linear

    def forward(self, x):
        '''
            return logits_list(g)  in paper
        '''
        batch_size = x.shape[0]
        gru_inputs = []
        for index, linear in enumerate(self.linear_feature_list):
            gru_input = linear(x[:, index].float())
            gru_inputs.append(gru_input)
        # gru_inputs = self.linear(x)
        gru_inputs = self.leaky_relu(torch.stack(gru_inputs))
        gru_inputs = torch.transpose(self.batch_norm(torch.transpose(gru_inputs,1,2)),1,2)
        outputs, _ = self._compute_gru(self.gru, gru_inputs, batch_size)
        logit_list = []
        logit_discriminator = []
        for index, (t_fw, t_bw) in enumerate(zip(self.output_points_fw, self.output_points_bw)):
            gru_output = []
            if t_fw is not None and not math.isnan(t_fw):
                gru_output.append(outputs[int(t_fw), :, :self.hidden_units]) 
            if t_bw is not None and not math.isnan(t_bw):
                gru_output.append(outputs[int(t_bw), :, self.hidden_units:])
            gru_output = torch.cat(gru_output, dim=1).to(self.device)
            logit = self.list_linear[index](gru_output)
            logit_list.append(logit)
        return logit_list, logit_discriminator

    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        hidden = weight.new(2, batch, self.hidden_units).zero_()
        return hidden
    
    def _compute_gru(self, GRUs, _input, batch_size):
        hidden = self.init_hidden(batch_size)
        for i, gru in enumerate(GRUs):
            output, state = gru(_input, hidden)
            output = torch.transpose(self.batch_norm_list[i](torch.transpose(output,1,2)),1,2)
            if i > 0:
                _input = _input + output
            else:
                _input = output
            hidden = state
        _input = self.leaky_relu(_input)
        logits, state = _input, hidden
        return logits, state


class EncoderGRU(nn.Module):
    def __init__(self, model_config, device):
        super(EncoderGRU, self).__init__()
        self.input_dim = model_config['input_dim']
        self.hidden_units = model_config['num_units']
        self.num_classes = model_config['num_classes']
        self.num_outputs = model_config['num_outputs']
        self.feature_size = model_config['feature_size']
        self.num_inputs = model_config['num_inputs']
        self.output_points_fw = model_config['output_points_fw']
        self.output_points_bw = model_config['output_points_bw']
        self.num_layers = model_config['num_layers']
        self.device = device
        self.linear = nn.Linear(self.input_dim, self.feature_size)
        self.gru = nn.GRU(self.feature_size, self.hidden_units, num_layers=self.num_layers, batch_first=True)

    def forward(self, _input, hidden):
        _input = _input.reshape(_input.shape[0], 1, -1)
        # print(_input.shape)
        logit = self.linear(_input)
        logit, state = self.gru(logit, hidden)
        return logit, state

    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch, self.hidden_units).zero_()
        return hidden

class DecoderGRU(nn.Module):
    def __init__(self, model_config, device):
        super(DecoderGRU, self).__init__()
        self.input_dim = model_config['input_dim']
        self.hidden_units = model_config['num_units']
        self.num_classes = model_config['num_classes']
        self.num_outputs = model_config['num_outputs']
        self.feature_size = model_config['feature_size']
        self.num_inputs = model_config['num_inputs']
        self.output_points_fw = model_config['output_points_fw']
        self.output_points_bw = model_config['output_points_bw']
        self.region = model_config['region']
        self.num_layers = model_config['num_layers']
        self.dropout_p = 0.1

        self.device = device
        self.embedding = nn.Embedding(self.num_outputs, self.hidden_units)
        self.gru = nn.GRU(self.hidden_units, self.hidden_units, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_units, self.num_classes)

        self.attn = nn.Linear(self.hidden_units * 2, self.num_inputs)
        self.attn_combine = nn.Linear(self.hidden_units * 2, self.hidden_units)
        self.dropout = nn.Dropout(self.dropout_p)


    def forward(self, _input, hidden, encoder_outputs):
        embedded = self.embedding(_input).view(1, _input.shape[0], -1)
        embedded = self.dropout(embedded)
        tmp = torch.cat((embedded[0], hidden[0]), 1)
        attn_weights = F.softmax(
            self.attn(tmp), dim=-1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)
        output = torch.cat((embedded[0], attn_applied[:,0]), 1)
        output = self.attn_combine(output).unsqueeze(0)


        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        logit = self.out(output)
        # output = torch.sigmoid(output)
        return logit, hidden, output

