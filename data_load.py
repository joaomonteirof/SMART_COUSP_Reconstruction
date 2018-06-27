import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class Loader(Dataset):

	def __init__(self, input_file_name, output_file_name):
		super(Loader, self).__init__()
		self.input_file_name = input_file_name
		self.output_file_name = output_file_name
		self.in_file = None
		self.out_file = None

		open_file = h5py.File(output_file_name, 'r')

		self.len = open_file['data'].shape[0]

		open_file.close()

	def __getitem__(self, index):

		if not self.in_file: self.in_file = h5py.File(self.input_file_name, 'r')
		in_samples = self.in_file['input_samples'][:,:,index].squeeze().T
		in_samples = torch.from_numpy(in_samples).float().unsqueeze(0)

		self.in_file.close()

		if not self.out_file: self.out_file = h5py.File(self.output_file_name, 'r')
		out_samples = (torch.from_numpy(self.out_file['data'][index]).float().view([40, 30, 30]) - 0.5) / 0.5

		return in_samples, out_samples

	def __len__(self):
		return self.len

class Loader_manyfiles(Dataset):

	def __init__(self, input_file_base_name, output_file_base_name, n_files):
		super(Loader_manyfiles, self).__init__()
		self.in_file_base_name = input_file_base_name
		self.out_file_base_name = output_file_base_name

		open_file = h5py.File(output_file_base_name+'_1.hdf', 'r')

		self.len_per_file = open_file['data'].shape[0]
		self.total_len = n_files*self.len_per_file

		open_file.close()

	def __getitem__(self, index):

		file_ = index // self.len_per_file + 1
		index = index % self.len_per_file

		in_file = h5py.File(self.in_file_base_name+'_'+str(file_)+'.hdf', 'r')
		in_samples = self.in_file['input_samples'][:,:,index].squeeze().T
		in_samples = torch.from_numpy(in_samples).float().unsqueeze(0)
		in_file.close()

		out_file = h5py.File(self.out_file_base_name+'_'+str(file_)+'.hdf', 'r')
		out_samples = (torch.from_numpy(self.out_file['data'][index]).float().view([40, 30, 30]) - 0.5) / 0.5
		out_file.close()

		return in_samples, out_samples

	def __len__(self):
		return self.total_len
