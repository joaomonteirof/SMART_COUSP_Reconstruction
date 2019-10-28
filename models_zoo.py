import torch
import torch.nn as nn
import torch.nn.functional as F

class model_gen(nn.Module):
	def __init__(self, cuda_mode):
		super(model_gen, self).__init__()

		self.cuda_mode = cuda_mode

		## Assuming (200, 299) inputs

		self.features = nn.Sequential(
			nn.Conv2d(1, 1024, kernel_size=(5,5), padding=(2,1), stride=(2,2), bias=False),
			nn.BatchNorm2d(1024),
			nn.ReLU(),
			nn.Conv2d(1024, 512, kernel_size=(5,5), padding=(2,1), stride=(2,2), bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512, 256, kernel_size=(5,5), padding=(2,1), stride=(2,2), bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 128, kernel_size=(5,5), padding=(2,1), stride=(2,2), bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 100, kernel_size=(5,5), padding=(2,0), stride=(2,2), bias=False),
			nn.BatchNorm2d(100),
			nn.ReLU() )

		self.lstm = nn.LSTM(49, 256, 2, bidirectional=True, batch_first=False)

		self.fc = nn.Linear(256*2, 100)

	def forward(self, x):

		x = self.features(x).squeeze(1).transpose(1,0)
		x = x.view(x.size(0), x.size(1), -1)

		batch_size = x.size(1)
		seq_size = x.size(0)

		h0 = torch.zeros(4, batch_size, 256)
		c0 = torch.zeros(4, batch_size, 256)

		if self.cuda_mode:
			h0 = h0.cuda()
			c0 = c0.cuda()

		x, h_c = self.lstm(x, (h0, c0))

		x = torch.tanh( self.fc( x ) )

		return x.transpose(0,1)
