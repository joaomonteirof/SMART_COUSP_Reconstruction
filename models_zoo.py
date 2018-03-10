import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Variable

class model(nn.Module):
	def __init__(self, cuda_mode):
		super(model, self).__init__()

		self.cuda_mode = cuda_mode

		## Considering (30, 90) inputs

		self.features = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(3,11), padding=1, stride=(1,2), bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.Conv2d(16, 32, kernel_size=(3,9), padding=1, stride=(2,2), bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=(3,5), padding=1, stride=(2,2), bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 128, kernel_size=5, padding=2, stride=2, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 96, kernel_size=5, padding=2, output_padding=1, stride=2, bias=False),
			nn.BatchNorm2d(96),
			nn.ReLU(),
			nn.ConvTranspose2d(96, 64, kernel_size=3, padding=1, stride=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 40, kernel_size=3, padding=1, stride=1, bias=False),
			nn.BatchNorm2d(40),
			nn.ReLU() )

		self.lstm_1 = nn.LSTM(30*30, 30*30, 1, bidirectional=True, batch_first=False)

		self.lstm_2 = nn.LSTM(2*30*30, 30*30, 1, bidirectional=True, batch_first=False)

		self.fc = nn.Linear(2*30*30,30*30)


	def forward(self, x):
		x = self.features(x)

		x = x.view(x.size(1), x.size(0), -1)

		batch_size = x.size(1)
		seq_size = x.size(0)

		h0 = Variable(torch.zeros(2, batch_size, 30*30))
		c0 = Variable(torch.zeros(2, batch_size, 30*30))

		if self.cuda_mode:
			h0 = h0.cuda()
			c0 = c0.cuda()

		x, h_c = self.lstm_1(x, (h0, c0))

		x, _ = self.lstm_2(x, h_c)

		x = F.relu( self.fc( x.view(batch_size*seq_size, -1) ) )

		return x.view(batch_size, seq_size, -1)

		'''

		out = []

		for i in range(x.size(1)):

			out.append(F.relu(self.fc(x[:,i,:])))

		out = torch.stack(out)

		out = out.view(out.size(1), out.size(0), -1)

		return out

		'''

class small_model(nn.Module):
	def __init__(self, cuda_mode):
		super(small_model, self).__init__()

		self.cuda_mode = cuda_mode

		## Considering (30, 90) inputs

		self.features = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(3,7), padding=(1,0), stride=(1,2), bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.Conv2d(16, 32, kernel_size=(3,7), padding=(1,0), stride=(1,1), bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 40, kernel_size=(3,7), padding=(1,0), stride=(1,1), bias=False),
			nn.BatchNorm2d(40),
			nn.ReLU() )

		self.lstm_1 = nn.LSTM(30*30, 30*30, 1, bidirectional=True, batch_first=False)

		self.lstm_2 = nn.LSTM(2*30*30, 30*30, 1, bidirectional=True, batch_first=False)

		self.fc = nn.Linear(2*30*30,30*30)


	def forward(self, x):
		x = self.features(x)

		x = x.view(x.size(1), x.size(0), -1)

		batch_size = x.size(1)
		seq_size = x.size(0)

		h0 = Variable(torch.zeros(2, batch_size, 30*30))
		c0 = Variable(torch.zeros(2, batch_size, 30*30))

		if self.cuda_mode:
			h0 = h0.cuda()
			c0 = c0.cuda()

		x, h_c = self.lstm_1(x, (h0, c0))

		x, _ = self.lstm_2(x, h_c)

		x = F.relu( self.fc( x.view(batch_size*seq_size, -1) ) )

		return x.view(batch_size, seq_size, -1)

class discriminator(nn.Module):
	def __init__(self):
		super(discriminator, self).__init__()

		self.features = nn.Sequential(
			nn.Conv2d(40, 64, kernel_size=3, padding=1, stride=2, bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.1),
			nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.1),
			nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.1) )

		self.fc = nn.Linear(128*4*4, 1)

	def forward(self, x):

		## Considering (40, 30, 30) inputs

		x = x.view(x.size(0), 40, 30, 30)

		x = self.features(x)

		x = x.view(x.size(0), -1)

		x = self.fc(x)

		return F.sigmoid(x)

class encoder(nn.Module):
	def __init__(self, cuda_mode):
		super(encoder, self).__init__()

		self.cuda_mode = cuda_mode

		## Considering (30, 90) inputs

		self.features = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(3,7), padding=(1,0), stride=(1,2), bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.Conv2d(16, 32, kernel_size=(3,7), padding=(1,0), stride=(1,1), bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 40, kernel_size=(3,7), padding=(1,0), stride=(1,1), bias=False),
			nn.BatchNorm2d(40),
			nn.ReLU() )

		self.lstm_1 = nn.LSTM(30*30, 30*30, 1, bidirectional=False, batch_first=False)

	def forward(self, x):
		x = self.features(x)

		x = x.view(x.size(1), x.size(0), -1)

		batch_size = x.size(1)
		seq_size = x.size(0)

		h0 = Variable(torch.zeros(1, batch_size, 30*30))
		c0 = Variable(torch.zeros(1, batch_size, 30*30))

		if self.cuda_mode:
			h0 = h0.cuda()
			c0 = c0.cuda()

		x, h_c = self.lstm_1(x, (h0, c0))

		return x, h_c

class decoder(nn.Module):
	def __init__(self):
		super(decoder, self).__init__()

		self.lstm_2 = nn.LSTM(30*30, 30*30, 1, bidirectional=False, batch_first=False)

		self.fc = nn.Linear(30*30,30*30)


	def forward(self, x, h):

		x = x.view(x.size(1), x.size(0), -1)

		batch_size = x.size(1)
		seq_size = x.size(0)

		x, _ = self.lstm_2(x, h)

		x = F.relu( self.fc( x.view(batch_size*seq_size, -1) ) )

		return x.view(batch_size, seq_size, -1)

class model_cnn3d(nn.Module):
	def __init__(self, cuda_mode):
		super(model_cnn3d, self).__init__()

		self.cuda_mode = cuda_mode

		## Considering (30, 90) inputs

		self.features = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(3,11), padding=1, stride=(1,2), bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.Conv2d(16, 32, kernel_size=(3,9), padding=1, stride=(2,2), bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=(3,5), padding=1, stride=(2,2), bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 128, kernel_size=5, padding=2, stride=2, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 96, kernel_size=5, padding=2, output_padding=1, stride=2, bias=False),
			nn.BatchNorm2d(96),
			nn.ReLU(),
			nn.ConvTranspose2d(96, 64, kernel_size=3, padding=1, stride=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 40, kernel_size=3, padding=1, stride=1, bias=False),
			nn.BatchNorm2d(40),
			nn.ReLU() )

		self.res1 = res_block(1, 10)
		self.res2 = res_block(1, 10)
		self.res3 = res_block(1, 10)

		self.out_conv = nn.Sequential( nn.ConvTranspose2d(40, 40, kernel_size=1, padding=0) )

	def forward(self, x):

		x = self.features(x).unsqueeze(1)

		x = self.res1(x)
		x = self.res2(x)
		x = self.res3(x)

		x = x.squeeze()

		x = self.out_conv(x)

		return x.view(x.size(0), x.size(1), -1)

class res_block(nn.Module):
	def __init__(self, in_channels, factor):
		super(res_block, self).__init__()
		# with batch norm instead of dropout
		self.res = nn.Sequential(
			nn.Conv3d(in_channels, in_channels*factor, kernel_size=3, padding=1, stride=1, bias=False),
			nn.BatchNorm3d(in_channels*factor),
			nn.ReLU(),
			nn.Conv3d(in_channels*factor, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
			nn.BatchNorm3d(in_channels) )

	def forward(self, x):
		return self.res(x) + x
