import torch
from torch.utils.data import Dataset
import random
import numpy as np
from data_prep.input_data_gen import *

class Loader(Dataset):

	def __init__(self, data_path, mask_path=None, add_noise=False):
		super(Loader, self).__init__()

		self.data = torch.load(data_path)
		self.n_frames = self.data.size(-1)
		self.add_noise = add_noise
		if mask_path:
			self.mask = np.load(mask_path)
		else:
			self.mask = None

	def __getitem__(self, index):

		out = self.data[index]

		inp = get_streaking_image(out.numpy(), self.mask)
		inp = torch.from_numpy(inp).unsqueeze(0).float().contiguous()

		if self.add_noise:
			if random.random()>0.5:
				inp += torch.randn_like(inp)*random.choice([1e-1, 1e-2, 1e-3])
				inp.clamp_(0.0, 1.0)

		out = out.squeeze().unsqueeze(0).float().contiguous()

		return inp, out

	def __len__(self):
		return self.data.size(0)

class Loader_gen(Dataset):

	def __init__(self, data_path):
		super(Loader_gen, self).__init__()

		self.data = torch.load(data_path)
		self.n_frames = self.data.size(-1)

	def __getitem__(self, index):

		out = self.data[index, :, :, random.randint(0, self.n_frames-1)].squeeze().unsqueeze(0).float().contiguous()

		return out

	def __len__(self):
		return self.data.size(0)