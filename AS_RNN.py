from typing import List
from torch.nn import Parameter
import torch
from torch import nn

class AntisymmetricRNNCell(torch.jit.ScriptModule):
    def __init__(self, input_dim, hidden_size, eps, gamma, init_W_std=1, bias = True):
        super(AntisymmetricRNNCell, self).__init__()
        
        #init Vh 
        normal_sampler_V = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([1/input_dim]))
        self.Vh_weight = nn.Parameter(normal_sampler_V.sample((hidden_size, input_dim))[..., 0])
        self.bias = bias
        #init W
        normal_sampler_W = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([init_W_std/hidden_size]))
        self.W = nn.Parameter(normal_sampler_W.sample((hidden_size, hidden_size))[..., 0])
        
        #init biases
        self.Vh_b_i = nn.Parameter(torch.zeros(hidden_size))
        self.Vh_b_h = nn.Parameter(torch.zeros(hidden_size))
        
        #init diffusion
        self.gamma_I = nn.Parameter(torch.eye(hidden_size, hidden_size)*gamma, requires_grad=False)
        
        self.eps = nn.Parameter(torch.Tensor([eps]), requires_grad=False)
        
        
    @torch.jit.script_method
    def forward(self, x, h):
        # (W - WT - gammaI)h
        WmWT_h = torch.matmul(h, (self.W - self.W.transpose(1, 0) - self.gamma_I)).squeeze()
        
        # Vhx + bh
        Vh_x = torch.matmul(self.Vh_weight, x.t()).t() + self.Vh_b_i + self.Vh_b_h
        
        # (W - WT - gammaI)h + Vhx + bh
        linear_transform = WmWT_h + Vh_x

        # tanh((W - WT - gammaI)h + Vhx + bh)
        f = torch.tanh(linear_transform)

        #eq. 12
        h = h + self.eps*f
        return h

class AntisymmetricRNN(torch.jit.ScriptModule):
	def __init__(self, input_size, hidden_size=32, eps=0.01, gamma=0.01, use_gating=False, init_W_std=1,
				is_cuda=True, batch_first = True, num_layers = 1, bias = True):
		super(AntisymmetricRNN, self).__init__()
		if use_gating:
			self.cell = AntisymmetricGatingRNNCell(input_size, n_units, eps, gamma, init_W_std)
		else:
			self.cell = AntisymmetricRNNCell(input_size, hidden_size, eps, gamma, init_W_std)
		self._all_weights = [['weight_h', 'weight_x', 'bias_h', 'bias_x','gamma_I', 'eps']]

		for name, param in zip(self._all_weights[0], self.parameters()):
			setattr(self, name, param)
		
		self.hidden_size = hidden_size
		self.bias =  bias
		self.batch_first = batch_first
		
	@torch.jit.script_method
	def forward(self, x, h):
		#T = x.shape[1]
		if self.batch_first:
			x_ = x.transpose(0,1).unbind(0)
		else:
			x_ = x.unbind(0)
		outputs = torch.jit.annotate(List[Tensor], [])
		#outputs = []
		for t in range(len(x_)):
			h = self.cell(x_[t], h)
			outputs += [h.squeeze()]
		out = torch.stack(outputs)
		if self.batch_first:
			out = out.transpose(0,1)
		return out, h

	@property
	def all_weights(self) -> List[Parameter]:
		return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]
		
	
	