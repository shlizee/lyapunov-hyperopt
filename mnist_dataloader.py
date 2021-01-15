import torch
import pickle
import unidecode

import gc
import math
import string
import sys
import os
import os.path as path
import torchvision

import pdb


def create_dataset(config):
   #Load Datasets
	if path.exists(f'{config.data_dir}/train_data.p'):
		train_dataset = torch.load(f'MNIST/datasets/val_data.p')
		val_dataset = torch.load(f'MNIST/datasets/val_data.p')
		test_dataset = torch.load(f'MNIST/datasets/test_data.p')
	else:
		train_dataset = torchvision.datasets.MNIST(config.data_dir, train=True, download=True,
							transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

		test_dataset = torchvision.datasets.MNIST(config.data_dir, train=False, download=True,
									 transform=torchvision.transforms.Compose([
									   torchvision.transforms.ToTensor(),
									   torchvision.transforms.Normalize(
										 (0.1307,), (0.3081,))
									 ]))

		train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])
		
		torch.save(train_dataset, f'{config.data_dir}/train_data.p')
		torch.save(val_dataset,  f'{config.data_dir}/val_data.p')
		torch.save(test_dataset, f'{config.data_dir}/test_data.p')

    #Create Dataloader Objects
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
	val_loader  = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=True)
	test_loader =  torch.utils.data.DataLoader(dataset= test_dataset, batch_size=config.batch_size,  shuffle=True)
	
	return {'train_set':train_dataset, 'val_set':val_dataset, 'test_set':test_dataset}