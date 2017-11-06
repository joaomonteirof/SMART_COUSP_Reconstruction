import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class Loader(Dataset):

	def __init__(self, input_file, output_file):
		super(Loader, self).__init__()
		self.in_file = input_file
		self.out_file = output_file

	def __getitem__(self, index):
		in_file = h5py.File(self.in_file, 'r')
		in_samples = in_file['input_samples'][:,:,:,index]
		in_samples = in_samples.reshape([in_samples.shape[2], in_samples.shape[1], in_samples.shape[0]])
		in_file.close()

		out_file = h5py.File(self.out_file, 'r')
		out_samples = out_file['data'][index]
		out_file.close()

		return torch.from_numpy(in_samples).float(), torch.from_numpy(out_samples).float()

	def __len__(self):
		open_file = h5py.File(self.out_file, 'r')
		return open_file['data'].shape[0]
