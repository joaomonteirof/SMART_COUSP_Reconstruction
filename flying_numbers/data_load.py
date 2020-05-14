import torch
from torch.utils.data import Dataset
import random
import numpy as np
from data_prep.input_data_gen import *

class Loader(Dataset):

	def __init__(self, data_path):
		super(Loader, self).__init__()

		self.data = torch.load(data_path)
		self.n_frames = self.data.size(-1)

	def __getitem__(self, index):

		out = self.data[index]

		inp = inp = get_streaking_image(out.numpy())
		inp = torch.from_numpy(inp).unsqueeze(0).float().contiguous()

		return inp, out

	def __len__(self):
		return self.data.size(0)

class Loader_gen(Dataset):

	def __init__(self, data_path):
		super(Loader, self).__init__()

		self.data = torch.load(data_path)
		self.n_frames = self.data.size(-1)

	def __getitem__(self, index):

		out = self.data[index, :, :, random.randint(0, self.n_frames-1)].squeeze().unsqueeze(0).float().contiguous()

		return out

	def __len__(self):
		return self.data.size(0)