import torch
from torch import optim, nn
from dataloader import *
import lyapunov_test as lyap
import pickle as pkl

from keras.utils import to_categorical

class DataConfig(object):
    def __init__(self, data_dir, batch_size, input_seq_length, target_seq_length, train_frac, val_frac, test_frac= 0.0, blank = False):
        """
        Creates data configuration object
        
        Args:
            data_dir: directory where source data can be found
            input_size: dimensionality of input data
            input_seq_length: length of sequences input into network
            target_seq_length: length of targets being predicted by network
            train_frac/val_frac/test_frac: train, val, and test splits for data (need add to 1)
        """
        
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.input_seq_length = input_seq_length
        self.target_seq_length = target_seq_length
        
        if (train_frac + val_frac + test_frac) != 1.0:
            print('Sum of train/val/test split must be 1.0, but sum was {%.1f}. Setting train/val/test split to 0.8/0.1/0.1'.format(train_frac + val_frac + test_frac))
            self.train_frac, self.val_frac, self.test_frac = (0.8, 0.1, 0.1)
        else:
            self.train_frac = train_frac
            self.val_frac = val_frac
            self.test_frac = test_frac
        
        if blank:
            self.input_size = len(pkl.load(open('{}/vocab.pkl'.format(data_dir), 'rb')))
        else:
            self.datasets = create_dataset(self)
            self.input_size = len(self.datasets['char_to_int'])

class TrainConfig(object):
    def __init__(self, model_dir, batch_size, max_epoch, optimizer, learning_rate, optim_params = {}, start_epoch = 0, scheduler = None, scheduler_params = {}):
        """
        Creates training configuration object
        
        Args:
            model_dir: location for trained models to be stored/accessed
            batch_size: batch size to be fed into network
            max_epoch: maximum number of epochs to train network
            optimizer: optimizer name, to refer to one of the optimzers in torch.optim
                Choose from ['adadelta', 'adagrad', 'adam', 'adamW', 'sparseAdam', 'adamax', 'asgd', 'rmsProp', 'rprop', 'sgd']
            learning_rate: learning rate of optimizer
            optim_params: dictionary of additional paramaters for optimizer defined above
            start_epoch: epoch on which to start/restart training (default =0, set to last completed epoch to begin training from previous spot)
            scheduler: lr_scheduler from torch.optim.lr_scheduler (use only method name)
            scheduler_params: parameters for scheduler defined above
        """
        
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.optim_params = optim_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        
        self.optimizer_options ={'adadelta' : optim.Adadelta, 'adagrad': optim.Adagrad, 'adam':optim.Adam, 'adamW': optim.AdamW, 'sparseAdam': optim.SparseAdam,
                    'adamax': optim.Adamax, 'asgd': optim.ASGD, 'rmsProp': optim.RMSprop, 'rprop': optim.Rprop, 'sgd': optim.SGD}
    
    def get_optimizer(self, model_params):
        return self.optimizer_options[self.optimizer](model_params, lr = self.learning_rate, **self.optim_params)
    
    def get_scheduler(self, opt):
        return self.scheduler(opt, **self.scheduler_params)

class ModelConfig(object):
    def __init__(self, model_type, num_layers, hidden_size, input_size, dropout, init_type, init_params, device, bias = True, nonlinearity = 'tanh', id_init_param= None, batch_first = True):
        """
        Creates configuration for model hyperparameters
        
        Args:
            model_type: recurrent layer architecture (currently supported: 'rnn', 'lstm', 'gru')
            num_layers: number of stacked recurrent layers
            hidden_size: number of hidden units in each recurrent layer (assuming, for now, that all layers have same number of units)
            input_size: dimension of input (should match value in corresponding DataConfig)
            dropout: dropout value to be used in dropout layer following recurrent layers [0.0 - 1.0)
            init_type: Type of initialization used - choose from: 'uni', 'xav', CHECK WHICH ARE SUPPORTED
            init_param: dictionary of initialization parameters to be used in defining initialization above
            device: from torch.device, should be torch.device('cpu') or torch.device('cuda')
            id_param: initialization parameter key used to identify initialization in file name
        """
        self.rnn_atts = {'num_layers':num_layers, 'input_size': input_size, 'hidden_size':hidden_size, 
                    'bias':bias, 'batch_first':batch_first} 
        
        self.dropout = dropout
        self.model_type = model_type
        self.nonlinearity = nonlinearity
        self.init_type = init_type
        self.init_params = init_params
        self.device = device
        self.id_init_param = id_init_param
        
        if model_type == 'rnn':
            rnn_atts['nonlinearity'] = self.nonlinearity
        
        self.initializations ={'uni': nn.init.uniform_, 'normal': nn.init.normal_, 'xav_uni': nn.init.xavier_uniform_, 'xav_normal': nn.init.xavier_normal_,
                        'ortho': nn.init.orthogonal_, 'kai_uni': nn.init.kaiming_uniform_, 'kai_normal': nn.init.kaiming_normal_}
        
        self.rec_layers = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
        self.gate_sizes = {'rnn': hidden_size, 'lstm': 4*hidden_size, 'gru': 3*hidden_size}
    
    def get_init(self, tensor):
        return self.initializations[self.init_type](tensor, **self.init_params)
    
    def get_RNN(self):
        return self.rec_layers[self.model_type](**self.rnn_atts).to(self.device)
    
    def get_gate_size(self):
        return self.gate_sizes[self.model_type]


class FullConfig(object):
    def __init__(self, dataCon, trainCon, modelCon):
        self.data = dataCon
        self.train = trainCon
        self.model = modelCon
        self.device = modelCon.device

    #Return full file prefix name with identifying parameters for this experiment
    def name(self):
        return '{0}_L{1}_H{2}_{3}_drop{4}_{5}_lr{6}_{7}_{8}'.format(self.model.model_type, self.model.rnn_atts['num_layers'], self.model.rnn_atts['hidden_size'],
                                                                                self.model.nonlinearity, self.model.dropout, self.train.optimizer,
                                                                                self.train.learning_rate, self.model.init_type, self.model.init_params[self.model.id_init_param])
    
class LyapConfig(object):
	def __init__(self,batch_size, seq_length, ON_step = 1, warmup = 0, one_hot= False):
		self.batch_size = batch_size
		self.seq_length = seq_length
		self.T_ONS =ON_step
		self.warmup = warmup
		self.one_hot = one_hot

		self.input = None

	def get_input(self, fcon):
		xt, _ = fcon.data.datasets['test_set']
		if self.one_hot:    
			xt = torch.flatten(xt)
		else:
			xt = torch.FloatTensor(xt)
		data_length = xt.shape[0]
		while self.batch_size*self.seq_length > xt.shape[0]:
			xt = torch.cat((xt,xt), dim = 0)
	   
		#random shift of input data
		i = torch.randint(low = 0, high = data_length, size = (1,)).item()
		xt = torch.roll(xt, shifts = i, dims = 0)
		xt = xt[:self.batch_size *self.seq_length * math.floor(xt.shape[0]/(self.batch_size * self.seq_length))]
		if self.one_hot:
			xt = torch.from_numpy(to_categorical(xt.view(-1, self.batch_size, self.seq_length), fcon.model.rnn_atts['input_size']))
		self.input = xt
		return xt.to(fcon.device)
    
	def calc_lyap(self, le_input, model, fcon):
		model.eval()
		model.lyapunov = True
		h = model.init_hidden(self.batch_size)
		i = torch.randint(low = 0, high = le_input.shape[0], size =  (1,)).item()
		LEs, rvals = lyap.calc_LEs_an(le_input[i], h, model = model, k_LE = 10000, rec_layer = fcon.model.model_type, warmup = self.warmup, T_ons = self.T_ONS)
		LE_mean, LE_std = lyap.LE_stats(LEs)
		model.lyapunov = False
		
		return (LE_mean, LE_std), rvals