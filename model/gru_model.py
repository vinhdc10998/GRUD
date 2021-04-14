import torch
from torch import nn

class GRUModel(nn.Module):
    def __init__(self, model_config, type_model):
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
        self.type_model = type_model

        output_feature = self.feature_size*2
        self.features_1 = nn.Linear(self.input_dim, self.feature_size)
        self.features_2 = nn.Linear(self.feature_size, output_feature)

        self.gru_fw = nn.GRU(
            input_size = output_feature,
            hidden_size = self.hidden_units,
            num_layers = self.num_layers
        )
        
        self.gru_bw = nn.GRU(
            input_size = output_feature,
            hidden_size = self.hidden_units,
            num_layers = self.num_layers
        )
        
        self.list_linear = []
        for (t_fw, t_bw) in (zip(self.output_points_fw, self.output_points_bw)):
            if (t_fw is not None) and (t_bw is not None):
                self.list_linear.append(nn.Linear(self.hidden_units*2, self.num_classes, bias=True)) 
            else:
                self.list_linear.append(nn.Linear(self.hidden_units, self.num_classes, bias=True))
        self.list_linear = nn.ModuleList(self.list_linear)

    def _compute_gru(self, gru_cell, _input, hidden):
        if self.type_model == 'Higher':
            logits, state = gru_cell(_input, hidden)
            return logits, state
        elif self.type_model == 'Lower':
            pass


    def forward(self, x, hidden=None):
        '''
            return logits_list(g)  in paper
        '''
        _input = torch.unbind(x, dim=1)
        gru_inputs = []
        for index in range(self.num_inputs):
            gru_input = self.features_1(_input[index])
            gru_input = self.features_2(gru_input)
            gru_inputs.append(gru_input)


        fw_end = self.output_points_fw[-1]
        bw_start = self.output_points_bw[0]

        outputs_fw = [None] * self.num_inputs
        outputs_bw = [None] * self.num_inputs

        if fw_end is not None:
            inputs_fw = torch.stack(gru_inputs[: fw_end + 1])
            outputs, _ = self._compute_gru(self.gru_fw, inputs_fw, hidden)
            for t in range(fw_end + 1):
                outputs_fw[t] = outputs[t]

        if bw_start is not None:
            inputs_bw = torch.stack([
                gru_inputs[i]
                for i in range(self.num_inputs - 1, bw_start - 1, -1)
            ])
            outputs, _ = self._compute_gru(self.gru_bw, inputs_bw, hidden)
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
            logit_list.append(logit)
        return logit_list

    def init_hidden(self, number_of_variants):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, number_of_variants, self.hidden_units).zero_() # self.num_layers*2(bidirection)
        return hidden
    