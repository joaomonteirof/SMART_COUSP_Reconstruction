from __future__ import print_function
import argparse
import torch
import numpy as np
import os


if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Split train and test data')
	parser.add_argument('--data-path', type=str, default=None, required=True, metavar='Path', help='Path to input data. Can be downloaded from http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy')
	parser.add_argument('--n-train', type=int, default=5000, metavar='N', help='Number of train examples')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output results to')
	args = parser.parse_args()

	data = np.load(args.data_path)
	data = torch.from_numpy(data).float()/255.0
	data = data.transpose(0,1).transpose(1,-1)
	print(data.size())

	assert args.n_train < data.size(0), 'Number of train examples has to be smaller than the total number of examples. n-train: {}, n examples: {}'.format(args.n_train, data.size(0))

	idx = np.arange(data.size(0))
	np.random.shuffle(idx)

	idx_train, idx_test = idx[:args.n_train], idx[args.n_train:]

	data_train, data_test = data[idx_train,...], data[idx_test,...]

	basename = os.path.basename(args.data_path).split('.')[0]

	train_file_name, test_file_name = os.path.join(args.out_path, basename+'_train.p'), os.path.join(args.out_path, basename+'_test.p')

	torch.save(data_train, train_file_name)
	print('\nSaved {} with shape {}'.format(train_file_name, data_train.size()))
	torch.save(data_test, test_file_name)
	print('\nSaved: {} with shape {}\n'.format(test_file_name, data_test.size()))