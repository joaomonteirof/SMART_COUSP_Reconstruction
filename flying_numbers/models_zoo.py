import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class model_gen(nn.Module):
	def __init__(self, n_frames, cuda_mode):
		super(model_gen, self).__init__()

		self.cuda_mode = cuda_mode

		## Assuming (64, 83) inputs

		self.features = nn.Sequential(
			nn.Conv2d(1, 128, kernel_size=(5,5), padding=(1,1), stride=(1,1), bias=True),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 64, kernel_size=(5,5), padding=(1,1), stride=(2,2), bias=True),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 32, kernel_size=(3,3), padding=(1,1), stride=(2,2), bias=True),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, n_frames, kernel_size=(3,3), padding=(1,1), stride=(1,1), bias=True),
			nn.BatchNorm2d(n_frames),
			nn.ReLU() )

		self.lstm = nn.LSTM(270, 256, 2, bidirectional=True, batch_first=False)

		self.fc = nn.Linear(256*2, 256)

	def forward(self, x):

		x = self.features(x).squeeze(1).transpose(1,0)
		x = x.view(x.size(0), x.size(1), -1)

		batch_size = x.size(1)
		seq_size = x.size(0)

		h0 = torch.zeros(2*2, batch_size, 256)
		c0 = torch.zeros(2*2, batch_size, 256)

		if self.cuda_mode:
			h0 = h0.cuda()
			c0 = c0.cuda()

		x, h_c = self.lstm(x, (h0, c0))

		x = torch.tanh( self.fc( x ) )

		return x.transpose(0,1)
