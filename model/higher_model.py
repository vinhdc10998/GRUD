import torch
from torch import nn
from torch.autograd import Variable

class HigherModel(nn.Module):
    def __init__(self, model_config, batch_size=1, bidir=True):
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
        self.batch_size = batch_size

        self.features_ = torch.zeros((self.num_inputs, 2, self.feature_size))
        self.gru = nn.GRU(
            input_size = self.feature_size,
            hidden_size = self.hidden_units,
            num_layers = self.num_layers,
            bidirectional = self.bidir)
        
        self.list_linear = []
        for (t_fw, t_bw) in (zip(self.output_points_fw, self.output_points_bw)):
            if t_fw is not None and t_bw is not None:
                self.list_linear.append(nn.Linear(self.hidden_units*2, self.num_classes, bias=True)) 
            else:
                self.list_linear.append(nn.Linear(self.hidden_units, self.num_classes, bias=True))
        self.list_linear = nn.ModuleList(self.list_linear)


    def forward(self, input_, hidden=None):
        '''
            return logits_list(g)  in paper
        '''
        batch = input_.shape[0]
        gru_inputs = []
        for i in range(self.num_inputs):
            gru_input = torch.matmul(input_[:,i], self.features_[i])
            gru_inputs.append(gru_input)
        input_ = torch.reshape(torch.stack(gru_inputs),(batch, self.num_inputs,-1))
        logits, state = self.gru(input_, hidden)
        outputs_fw = logits[:,:, :self.hidden_units]
        outputs_bw = logits[:,:, self.hidden_units:]

        logit_list = []
        for index, (t_fw, t_bw) in enumerate(zip(self.output_points_fw, self.output_points_bw)):
            gru_output = []
            if t_fw is not None:
                gru_output.append(outputs_fw[:,t_fw,:]) 
            if t_bw is not None:
                gru_output.append(outputs_bw[:,t_bw,:])
            gru_output = torch.reshape(torch.stack(gru_output),(batch,1,-1))
            logit = self.list_linear[index](gru_output)
            logit_list.append(logit)

        return torch.stack(logit_list), state

    def init_hidden(self, number_of_variants):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers*2 , number_of_variants, self.hidden_units).zero_() # self.num_layers*2(bidirection)
        return hidden
    