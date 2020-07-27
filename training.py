import torch
from config import *
from models import RNNModel
from torch import nn
import dataloader as dl

import os

def load_checkpoint(full_con, load_epoch, verbose = False):
    if verbose:
        print("Train Directory:", full_con.train.model_dir)
    device = full_con.model.device
    model = RNNModel(full_con.model).to(device)
    optimizer = full_con.train.get_optimizer(model.parameters())
    ckpt_name = '{}/{}_e{}.ckpt'.format(full_con.train.model_dir, full_con.name(), load_epoch)
    if load_epoch > 0:
        if os.path.isfile(ckpt_name):
            ckpt = torch.load(ckpt_name)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            train_loss = ckpt['train_loss']
            val_loss = ckpt['val_loss']
        else:
            print("Expected file name {}".format(ckpt_name))
            raise ValueError("Asked to load checkpoint at epoch {0}, but checkpoint does not exist.".format(load_epoch))
    else:
        if verbose:
            print("Epoch = 0. Creating new checkpoint for untrained model")
        train_loss = []
        val_loss = 0.0
        ckpt = {'model_state_dict': model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(), 'train_loss':train_loss, 'val_loss':val_loss}
        torch.save(ckpt, ckpt_name)
    return model, optimizer, train_loss, val_loss

def save_checkpoint(full_con, model, optimizer, train_loss, val_loss, save_epoch):
    ckpt_name = '{}/{}_e{}.ckpt'.format(full_con.train.model_dir, full_con.name(), save_epoch)
    ckpt = {'model_state_dict': model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(), 'train_loss':train_loss, 'val_loss':val_loss}
    torch.save(ckpt, ckpt_name)

def train_model(full_con, model, optimizer, start_epoch= 0, print_interval = 1, save_interval = 1, verbose = True):
    device = full_con.device
    #ckpt = load_checkpoint(full_con, start_epoch)
    model, optimizer, train_loss, _ = load_checkpoint(full_con, start_epoch, verbose)
    #model.load_state_dict(ckpt['model_state_dict'])
    # optimizer.load_state_dict(ckpt['model_state_dict'])
    # train_loss = ckpt['loss']
    criterion = nn.CrossEntropyLoss()
    scheduler = full_con.train.get_scheduler(optimizer)

#     data = dl.create_dataset(full_con.data)
    train_input, train_target = (full_con.data.datasets['train_set'][0].to(device), full_con.data.datasets['train_set'][1].to(device))
    val_input, val_target = (full_con.data.datasets['val_set'][0].to(device), full_con.data.datasets['val_set'][1].to(device))
    if verbose:
        print('Training ...')
    
    for epoch in range(start_epoch+1, full_con.train.max_epoch+1):
        if epoch%print_interval == 0 and verbose:
            print('Training epoch {} of {}'.format(epoch, full_con.train.max_epoch), end = '')
        running_loss = 0.0
        hidden = model.init_hidden(full_con.train.batch_size)

        #train for all batches in the training set
        model.train()
        for batch_in, batch_target in zip(train_input, train_target):
            optimizer.zero_grad()
            loss = 0.0
            batch_out, _ = model(batch_in, hidden)
            loss += criterion(batch_out.view(-1, full_con.model.rnn_atts['input_size']), batch_target.view(-1,)).to(device)
            loss.backward()
            optimizer.step()
#             print(loss.item())
            running_loss += loss.item()
        train_loss.append(running_loss/(train_input.shape[0]))

        #Find validation loss
        model.eval()
        val_loss = 0.0
        for batch_in, batch_target in zip(val_input, val_target):
            batch_out, _ = model(batch_in, hidden)
            loss = criterion(batch_out.view(-1, full_con.model.rnn_atts['input_size']), batch_target.view(-1,)).to(device)
#             print(loss.item())
            val_loss += loss.item()
        
        val_loss = val_loss/(val_input.shape[0])
        scheduler.step()
        
        if epoch%print_interval == 0 and verbose:
            print(', Training Loss: {:.4f}, Validation Loss {:.4f}'.format(train_loss[-1], val_loss))

        #save model checkpoint
        if epoch%save_interval == 0:
            save_checkpoint(full_con, model, optimizer, train_loss, val_loss, epoch)
            
    return train_loss, val_loss