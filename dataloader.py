import torch
import pickle
import unidecode

import gc
import math
import string
import sys
import os
import os.path as path

import pdb

# convert all characters in input_file into int
'''
vocab_file: store char_to_int and int_to_char dictionaries as a tuple
data_file: store converted int
'''
def text_to_tensor(input_file, vocab_file, data_file):
	print('Loading text file...')

	try:
		file = unidecode.unidecode(open(input_file).read())
	except IOError:
		print("Could not read file: " + input_file) # fail to open input text file
		sys.exit()

	vocab = ''
	all_chars = string.printable    # all printable characters in python
	for ch in all_chars:
		if ch in file:
			vocab += ch # select characters that appear in the file
	#     if 'ê' in file:
	sp_chs = ['ê','à', 'é', 'ä'] #'\ufeff', '“', '’', '—', '‘', '”', 'á', 'ë', 'í', 'À', 'ó', 'ú', 'è', 'î', 'ô', 'ç', 'â', 'ï', 'ý', 'ö', 'ü', 'Á', 'œ', 'É', 'æ']
	for ch in sp_chs:
		vocab += ch

	char_to_int = {}    # mapping from character to int
	for i in range(len(vocab)):
		char_to_int[vocab[i]] = i

	int_to_char = {}    # mapping from int to character
	for k, v in char_to_int.items():
		int_to_char[v] = k

	# construct a tensor with all data
	print('Putting data into tensor...')
	data = torch.ByteTensor(len(file))  # store in into 1D first, then rearrange
	cache_len = 10000
	with open(input_file, 'r') as f:
		currlen = 0
		while True:
			raw_data = f.read(cache_len)
			if not raw_data: # end of file
				break
			for i in range(len(raw_data)):
				data[currlen + i] = char_to_int[raw_data[i]]
			currlen += len(raw_data)
	f.close()

	# save vocabulary
	print('Saving ', vocab_file, '...')
	f = open(vocab_file, 'wb')
	pickle.dump((char_to_int, int_to_char), f)  # save a tuple
	f.close()
	
	# save data tensor
	print('Saving ', data_file, '...')
	f = open(data_file, 'wb')
	pickle.dump(data, f)
	f.close()

# create dataset: train, val and test
def create_dataset(config):
    # get file paths
    input_file = path.join(config.data_dir, 'input.txt')
    vocab_file = path.join(config.data_dir, 'vocab.pkl')
    data_file = path.join(config.data_dir, 'data.pkl')

    # fetch file attributes to determine if we need to return preprocessing
    run_preproc = False
    if not (path.exists(vocab_file) and path.exists(data_file)):
        # prepro files do not exists, generate them
        print('vocab.pkl and data.pkl do not exists. Running preprocessing...')
        run_preproc = True
    else:
        # check if the input file was modifed since the last time we ran preporoc.
        # If so, we have to re-run the preprocessing
        input_attr = os.stat(input_file)
        vocab_attr = os.stat(vocab_file)
        data_attr = os.stat(data_file)
        if input_attr.st_mtime > vocab_attr.st_mtime or input_attr.st_mtime > data_attr.st_mtime:
            print('vocab.pkl or data.pkl detected as stale. Re-running preprocessing...')
            run_preproc = True

    if run_preproc:
        # construct a tensor with all the data, and vocab file
        print('One-time setup: preprocessing input text file ', input_file, '...')
        text_to_tensor(input_file, vocab_file, data_file)

    # load data tensor
    print('Loading ', data_file, '...')
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    f.close()

    # load dictionaries
    print(print('Loading ', vocab_file, '...'))
    with open(vocab_file, 'rb') as f:
        (char_to_int, int_to_char) = pickle.load(f)
    f.close()

    batch_size = config.batch_size
    seq_length = config.input_seq_length
    # cutoff the end of the file so that it divides evenly
    length = data.shape[0]  # number of characters
    if length % (batch_size * seq_length) != 0:
        print('Cutting off end of data so that it divides evenly')
        data = data[0: batch_size * seq_length * math.floor(length / (batch_size * seq_length))]

    # perform safety checks on split_fractions
    test_frac = max(0, 1 - (config.train_frac + config.val_frac))
    split_fractions = {}
    split_fractions['train_frac'] = config.train_frac
    split_fractions['val_frac'] = config.val_frac
    split_fractions['test_frac'] = test_frac
    assert split_fractions['train_frac'] >= 0 and split_fractions['train_frac'] <= 1, \
        'Bad split fraction ' + str(split_fractions['train_frac']) + ' for train, not between 0 and 1'
    assert split_fractions['val_frac'] >= 0 and split_fractions['val_frac'] <= 1, \
        'Bad split fraction ' + str(split_fractions['val_frac']) + ' for val, not between 0 and 1'
    assert split_fractions['test_frac'] >= 0 and split_fractions['test_frac'] <= 1, \
        'Bad split fraction ' + str(split_fractions['test_frac']) + ' for test, not between 0 and 1'

    # for each sequence, we generate a target: (a1, a2, ..., an-1, an) -> (a2, ..., an-1, an, a1)
    target = torch.ByteTensor(data.shape)
    for i in range(int(len(data) / seq_length)):
        src_seq = data[i * seq_length : (i + 1) * seq_length]   # length: seq_length
        dest_seq = src_seq.clone()
        dest_seq[0: -1] = src_seq[1:]   # '-1' is not included
        dest_seq[-1] = src_seq[0]
        target[i * seq_length : (i + 1) * seq_length] = dest_seq

    train_batches = math.floor(split_fractions['train_frac'] * (len(data) / (batch_size * seq_length)))
    train_idx = 0   # start point of train set
    input_set = torch.LongTensor(train_batches, batch_size, seq_length)
    target_set = torch.LongTensor(train_batches, batch_size, seq_length)
    for i in range(train_batches):
        for j in range(batch_size):
            start_idx = train_idx + i * (batch_size * seq_length) + j * seq_length
            end_idx = start_idx + seq_length
            input_set[i][j] = data[start_idx : end_idx]
            target_set[i][j] = target[start_idx: end_idx]
    train_set = (input_set, target_set) # a tuple of two 3d tensors

    val_batches = math.floor(split_fractions['val_frac'] * (len(data) / (batch_size * seq_length)))
    val_idx = train_batches * batch_size * seq_length   # start point of validation set
    input_set = torch.LongTensor(val_batches, batch_size, seq_length)
    target_set = torch.LongTensor(val_batches, batch_size, seq_length)
    for i in range(val_batches):
        for j in range(batch_size):
            start_idx = val_idx + i * (batch_size * seq_length) + j * seq_length
            end_idx = start_idx + seq_length
            input_set[i][j] = data[start_idx : end_idx]
            target_set[i][j] = target[start_idx: end_idx]
    val_set = (input_set, target_set)   # a tuple of two 3d tensors

    test_batches = math.floor(split_fractions['test_frac'] * (len(data) / (batch_size * seq_length)))
    test_idx = (train_batches + val_batches) * batch_size * seq_length  # start point of test set
    input_set = torch.LongTensor(test_batches, batch_size, seq_length)
    target_set = torch.LongTensor(test_batches, batch_size, seq_length)
    for i in range(test_batches):
        for j in range(batch_size):
            start_idx = test_idx + i * (batch_size * seq_length) + j * seq_length
            end_idx = start_idx + seq_length
            input_set[i][j] = data[start_idx: end_idx]
            target_set[i][j] = target[start_idx: end_idx]
    test_set = (input_set, target_set)

    # a helpful warning
    if len(data) / (batch_size * seq_length) < 50:
        print('WARNING: than 50 batches in the data in total? Looks like very small dataset. '
              'You probably want to use smaller batch_size and/or seq_length.')

    # print a summary of dataset
    print('Data load done! Number of data batches in train: %d, val: %d, test: %d' %
          (train_batches, val_batches, test_batches))

    gc.collect()
	
    return {'train_set':train_set, 'val_set':val_set, 'test_set':test_set, 'char_to_int':char_to_int, 'int_to_char':int_to_char}