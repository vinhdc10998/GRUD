import torch 
from torch import nn

class LowerModel(nn.Module):
    def __init__(self, model_config, bidir=True):
        super(LowerModel,self).__init__()
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

        
        self.gru1 = nn.GRU(self.feature_size, self.hidden_units, bidirectional = self.bidir)
        self.gru2 = nn.GRU(self.hidden_units*2 + self.feature_size, self.hidden_units, bidirectional = self.bidir)
        self.gru3 = nn.GRU(self.hidden_units*2 + self.feature_size, self.hidden_units, bidirectional = self.bidir)
        self.gru4 = nn.GRU(self.hidden_units*2 + self.feature_size, self.hidden_units, bidirectional = self.bidir)

    def forward(self, input_, hidden=None):
        gru_inputs = []
        outputs_fw = [None] * self.num_inputs
        outputs_bw = [None] * self.num_inputs
        features = torch.zeros((self.num_inputs, 2, self.feature_size))

        for i in range(self.num_inputs):
            gru_input = torch.matmul(input_[0][i], features[i])
            gru_inputs.append(gru_input)
        input_ = torch.reshape(torch.stack(gru_inputs),(1, -1,self.feature_size))
        #Model
        output_1, state_1 = self.gru1(input_, hidden)
        output_1_residual = torch.cat((input_, output_1), dim=2)

        output_2, state_2 = self.gru2(output_1_residual, state_1)
        output_2_residual = torch.cat((output_1_residual, output_2), dim=2)

        output_3, state_3 = self.gru3(output_2_residual, state_2)
        output_3_residual = torch.cat((output_2_residual, output_3), dim=2)

        output_4, state_4 = self.gru4(output_3_residual, state_3)
        output_4_residual = torch.cat((output_3_residual, output_4), dim=2)
        print("[DEBUG]:", output_4_residual.size())
        pause = input("PAUSE...")

        outputs_fw = output_4_residual[0,:, :self.hidden_units]
        outputs_bw = output_4_residual[0,:, self.hidden_units:]

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

        return torch.stack(logit_list), state_4

    def init_hidden(self, number_of_variants):
        weight = next(self.parameters()).data
        hidden = weight.new(2 , number_of_variants, self.hidden_units).zero_() # self.num_layers*2(bidirection)
        return hidden
