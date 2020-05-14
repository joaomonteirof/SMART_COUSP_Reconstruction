import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class add_noise(nn.Module):
	def __init__(self, input_noise):
		super(add_noise, self).__init__()
		self.input_noise = input_noise
	def forward(self, x):

		if self.training and self.input_noise:
			with torch.no_grad():
				noise = torch.randn_like(x)*np.random.choice([1e-1, 1e-2, 1e-3, 1e-4])
				if noise.size(0)>1:
					noise[np.random.choice(np.arange(noise.size(0)), size=noise.size(0)//2, replace=False)] = 0.0
				x += noise
				x = torch.clamp(x, 0.0, 1.0)
		return x

class model_gen(nn.Module):
	def __init__(self, n_frames, cuda_mode, input_noise=False):
		super(model_gen, self).__init__()

		self.cuda_mode = cuda_mode

		## Assuming (256, 355) inputs

		self.noise_layer = add_noise(input_noise)

		self.features = nn.Sequential(
			nn.Conv2d(1, 512, kernel_size=(5,5), padding=(2,1), stride=(2,2), bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512, 256, kernel_size=(5,5), padding=(2,1), stride=(2,2), bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=(5,5), padding=(2,1), stride=(2,2), bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 128, kernel_size=(5,5), padding=(2,1), stride=(2,2), bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, n_frames, kernel_size=(5,5), padding=(2,1), stride=(2,2), bias=False),
			nn.BatchNorm2d(n_frames),
			nn.ReLU() )

		self.lstm = nn.LSTM(80, 128, 2, bidirectional=True, batch_first=False)

		self.fc = nn.Linear(128*2, 128)

	def forward(self, x):

		x = self.noise_layer(x)

		x = self.features(x).squeeze(1).transpose(1,0)
		x = x.view(x.size(0), x.size(1), -1)

		batch_size = x.size(1)
		seq_size = x.size(0)

		h0 = torch.zeros(4, batch_size, 128)
		c0 = torch.zeros(4, batch_size, 128)

		if self.cuda_mode:
			h0 = h0.cuda()
			c0 = c0.cuda()

		x, h_c = self.lstm(x, (h0, c0))

		x = torch.tanh( self.fc( x ) )

		return x.transpose(0,1)
