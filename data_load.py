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

		in_data = scipy.io.loadmat(self.in_file)
		out_data = scipy.io.loadmat(self.out_file)

		print(a.shape)
			
		return torch.from_numpy(in_data['Data'][0][index]).float(), torch.from_numpy(in_data['Data'][0][index]).float()

	def __len__(self):
		in_data = scipy.io.loadmat(self.in_file)
		return len(in_data['Data'][0])
