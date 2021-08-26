import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class model_gen(nn.Module):
	def __init__(self, n_frames, cuda_mode):
		super(model_gen, self).__init__()

		self.cuda_mode = cuda_mode

		## Assuming (64, 143) inputs

		self.features = nn.Sequential(
			nn.Conv2d(1, 512, kernel_size=(5,5), padding=(2,1), stride=(2,2), bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512, 256, kernel_size=(5,5), padding=(2,1), stride=(2,2), bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 128, kernel_size=(5,5), padding=(2,1), stride=(1,2), bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, n_frames, kernel_size=(5,5), padding=(2,1), stride=(1,1), bias=False),
			nn.BatchNorm2d(n_frames),
			nn.ReLU() )

		self.lstm = nn.LSTM(160, 256, 2, bidirectional=True, batch_first=False)

		self.fc = nn.Linear(256*2, 100)

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



class ResidualBlock(nn.Module):

	def __init__(self):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.conv2(out)
		out += x
		out = F.relu(out)
		return out

class model_baseline(nn.Module):
	def __init__(self, n_frames):
		super(model_baseline, self).__init__()

		self.input_conv = nn.Sequential(nn.Conv2d(n_frames, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU())

		self.intermediate_conv = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU())

		self.output_conv = nn.Conv2d(64, n_frames, kernel_size=1, stride=1, padding=0, bias=True)

		self.res_enc_1 = ResidualBlock()
		self.res_dec_1 = ResidualBlock()
		self.res_enc_2 = ResidualBlock()
		self.res_dec_2 = ResidualBlock()
		self.res_enc_3 = ResidualBlock()
		self.res_dec_3 = ResidualBlock()
		self.res_enc_4 = ResidualBlock()
		self.res_dec_4 = ResidualBlock()
		self.res_enc_5 = ResidualBlock()
		self.res_dec_5 = ResidualBlock()


	def forward(self, x):
		x = x.squeeze(1).transpose(1,-1)
		x_in = self.input_conv(x)
		x_enc_1 = self.res_enc_1(x_in)
		x_enc_2 = self.res_enc_2(x_enc_1)
		x_enc_3 = self.res_enc_3(x_enc_2)
		x_enc_4 = self.res_enc_4(x_enc_3)
		x_enc_5 = self.res_enc_5(x_enc_4)
		x_dec_0 = self.intermediate_conv(x_enc_5) + x_enc_5
		x_dec_1 = self.res_dec_1(x_dec_0)
		x_dec_2 = self.res_dec_2(x_dec_1+x_enc_4)
		x_dec_3 = self.res_dec_3(x_dec_2+x_enc_3)
		x_dec_4 = self.res_dec_4(x_dec_3+x_enc_2)
		x_dec_5 = self.res_dec_5(x_dec_4+x_enc_1)
		x_out = self.output_conv(x_dec_5)
		return x_out.transpose(1,-1).unsqueeze(1)
