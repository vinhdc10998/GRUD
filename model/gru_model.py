import numpy as np
import torch
from torch import nn

class GRUModel(nn.Module):
    def __init__(self, model_config, device, type_model):
        super(GRUModel,self).__init__()
        self.input_dim = model_config['input_dim']
        self.hidden_units = model_config['num_units']
        self.num_classes = model_config['num_classes']
        self.num_outputs = model_config['num_outputs']
        self.num_layers = model_config['num_layers']
        self.feature_size = model_config['feature_size']
        self.num_inputs = model_config['num_inputs']
        self.output_points_fw = model_config['output_points_fw']
        self.output_points_bw = model_config['output_points_bw']
        self.region = model_config['region']
        self.type_model = type_model
        # self._features = torch.tensor(np.load(f'model/features/region_{self.region}_model_features.npy')).to(device)

        output_features = self.feature_size*5
        self.features_1 = nn.Linear(self.num_classes, output_features)
        self.batch_norm_1 = nn.BatchNorm1d(output_features)
        self.batch_norm_2 = nn.BatchNorm1d(self.num_classes)
        self.sigmoid = nn.Sigmoid()

        self.gru = nn.ModuleDict(self._create_gru_cell(
            output_features, 
            self.hidden_units, 
            self.num_layers, 
            self.type_model
        ))
        
        self.list_linear = nn.ModuleList(self._create_linear_list(
            output_features,
            self.hidden_units,
            self.num_layers,
            self.num_classes,
            self.output_points_fw,
            self.output_points_bw,
            self.type_model
        ))

    @staticmethod
    def _create_gru_cell(input_size, hidden_units, num_layers, type_model, dropout=0.2):
        gru = None
        if type_model == 'Higher':
            gru = nn.GRU(
                    input_size = input_size,
                    hidden_size = hidden_units,
                    num_layers = num_layers,
                    dropout = dropout
                )
        elif type_model == 'Lower':
            input_size_1 = input_size
            input_size_2 = input_size_1 + hidden_units
            input_size_3 = input_size_2 + hidden_units
            input_size_4 = input_size_3 + hidden_units
            gru = nn.ModuleList([
                nn.GRU(input_size_1, hidden_units),
                nn.GRU(input_size_2, hidden_units),
                nn.GRU(input_size_3, hidden_units),
                nn.GRU(input_size_4, hidden_units)
            ])
        return {
            'fw': gru,
            'bw': gru
        }

    @staticmethod
    def _create_linear_list(feature_size, hidden_units, num_layers, num_classes, output_points_fw, output_points_bw, type_model):
        if type_model == 'Higher':
            input_size = hidden_units
        elif type_model == 'Lower':
            input_size = feature_size + hidden_units*num_layers
        list_linear = []
        for (t_fw, t_bw) in (zip(output_points_fw, output_points_bw)):
            if (t_fw is not None) and (t_bw is not None):
                list_linear.append(nn.Linear(input_size*2, num_classes, bias=True)) 
            else:
                list_linear.append(nn.Linear(input_size, num_classes, bias=True))
        return list_linear

    def _compute_gru(self, gru_cell, _input, hidden):
        if self.type_model == 'Higher':
            gru = gru_cell
            logits, state = gru(_input, hidden)
        elif self.type_model == 'Lower':
            for i, gru in enumerate(gru_cell):
                output, state = gru(_input, hidden)
                hidden = state
                _input = torch.cat((output, _input), dim=2)
            logits, state = _input, hidden
        return logits, state

    def forward(self, x, hidden=None):
        '''
            return logits_list(g)  in paper
        '''
        _input = torch.unbind(x, dim=1)
        gru_inputs = []
        for index in range(self.num_inputs):
            gru_input = self.features_1(_input[index])
            gru_input = self.batch_norm_1(gru_input)
            # gru_input = self.sigmoid(gru_input)
            gru_inputs.append(gru_input)

        fw_end = self.output_points_fw[-1]
        bw_start = self.output_points_bw[0]

        outputs_fw = [None for _ in range(self.num_inputs)]
        outputs_bw = [None for _ in range(self.num_inputs)]
        
        if fw_end is not None:
            inputs_fw = torch.stack(gru_inputs[: fw_end + 1])
            outputs, _ = self._compute_gru(self.gru['fw'], inputs_fw, hidden)
            for t in range(fw_end + 1):
                outputs_fw[t] = outputs[t]
        if bw_start is not None:
            inputs_bw = torch.stack([
                gru_inputs[i]
                for i in range(self.num_inputs - 1, bw_start - 1, -1)
            ])
            outputs, _ = self._compute_gru(self.gru['bw'], inputs_bw, hidden)
            for i, t in enumerate(
                range(self.num_inputs - 1, bw_start - 1, -1)):
                outputs_bw[t] = outputs[i]

        logit_list = []
        for index, (t_fw, t_bw) in enumerate(zip(self.output_points_fw, self.output_points_bw)):
            gru_output = []
            if t_fw is not None:
                gru_output.append(outputs_fw[t_fw]) 
            if t_bw is not None:
                gru_output.append(outputs_bw[t_bw])
            gru_output = torch.cat(gru_output, dim=1)
            logit = self.list_linear[index](gru_output)
            # logit = self.batch_norm_2(logit)
            logit = self.sigmoid(logit)
            logit_list.append(logit)
        return logit_list

    def init_hidden(self, batch):
        num_layers = self.num_layers
        if self.type_model == 'Lower':
            num_layers = 1
        weight = next(self.gru.parameters()).data
        hidden = weight.new(num_layers, batch, self.hidden_units).zero_() # self.num_layers*2(bidirection)
        return hidden
    