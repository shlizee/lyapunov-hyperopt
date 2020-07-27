import torch
from torch import nn
from config import ModelConfig
from keras.utils import to_categorical

class RNNModel(nn.ModuleList):
    def __init__(self, model_con):
        super(RNNModel, self).__init__()
        
        self.con = model_con
        self.L = self.con.rnn_atts['num_layers']*self.con.rnn_atts['hidden_size']
        
        self.encoder = lambda xt: torch.from_numpy(to_categorical(xt.cpu(), self.con.rnn_atts['input_size'])).to(self.con.device)
        self.dropout = nn.Dropout(p = self.con.dropout)
        self.fc = nn.Linear(in_features = self.con.rnn_atts['hidden_size'], out_features = self.con.rnn_atts['input_size'])
        
        self.rnn_layer = model_con.get_RNN()
        self.gate_size = model_con.get_gate_size()
        for layer_p in self.rnn_layer._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    self.con.get_init(self.rnn_layer.__getattr__(p))
        self.lyapunov = False
        
    def forward(self, xt, h):
        if self.lyapunov:
            encoded = torch.nn.Identity()(xt)
        else:
            encoded = self.encoder(xt)
           
        self.rnn_layer.flatten_parameters()
        rnn_out, rnn_hn = self.rnn_layer(encoded, h)
        d_out = self.dropout(rnn_out)
        output = self.fc(d_out)
        return output, rnn_hn

    def init_hidden(self, batch_size):
        h = torch.zeros(self.con.rnn_atts['num_layers'], batch_size, self.con.rnn_atts['hidden_size']).to(self.con.device)
        c = torch.zeros(self.con.rnn_atts['num_layers'], batch_size, self.con.rnn_atts['hidden_size']).to(self.con.device)
        
        if self.con.model_type == 'lstm':
            return (h,c)
        else:
            del c
            return h