import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Variable

class model(nn.Module):
	def __init__(self, cuda_mode):
		super(model, self).__init__()

		self.cuda_mode = cuda_mode

		## Considering (30,90) inputs

		self.features = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(3,11), padding=1, stride=(1,2)),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.Conv2d(16, 32, kernel_size=(3,9), padding=1, stride=(2,2)),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=(3,5), padding=1, stride=(2,2)),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=(3,3), padding=1, stride=(1,1)),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 128, kernel_size=5, padding=2, stride=2),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 96, kernel_size=5, padding=2, output_padding=1, stride=2),
			nn.BatchNorm2d(96),
			nn.ReLU(),
			nn.ConvTranspose2d(96, 64, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 40, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm2d(40),
			nn.ReLU() )

		self.lstm_1 = nn.LSTM(30*30, 30*15, 1, bidirectional=True, batch_first=True)

		self.lstm_2 = nn.LSTM(30*30, 30*15, 1, bidirectional=True, batch_first=True)

		self.fc = nn.Linear(30*30,30*30)


	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), x.size(1), -1)

		h0 = Variable(torch.zeros(2, x.size(0), 30*15))
		c0 = Variable(torch.zeros(2, x.size(0), 30*15))

		if self.cuda_mode:
			h0 = h0.cuda()
			c0 = c0.cuda()

		x, h_c = self.lstm_1(x, (h0, c0))

		x, _ = self.lstm_2(x, h_c)

		out = []

		for i in range(x.size(1)):

			out.append(F.sigmoid(self.fc(x[:,i,:])))

		out = torch.stack(out)

		out = out.view(out.size(1), out.size(0), -1)

		return out
