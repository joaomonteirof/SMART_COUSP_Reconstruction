import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
	def __init__(self):
		super(model, self).__init__()

		## Considering (32,1500) inputs

		self.features = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(3,25), padding=1, stride=(1,2)),
			nn.BatchNorm2d(16),
			nn.RELU(),
			nn.Conv2d(16, 32, kernel_size=(3,25), padding=1, stride=(2,4)),
			nn.BatchNorm2d(32),
			nn.RELU(),
			nn.Conv2d(32, 64, kernel_size=(3,25), padding=1, stride=(2,4)),
			nn.BatchNorm2d(64),
			nn.RELU(),
			nn.Conv2d(64, 128, kernel_size=(3,25), padding=(1,0), stride=(1,2)),
			nn.BatchNorm2d(128),
			nn.RELU(),
			nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm2d(128),
			nn.RELU(),
			nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2),
			nn.BatchNorm2d(64),
			nn.RELU(),
			nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2),
			nn.BatchNorm2d(128),
			nn.RELU(),
			nn.ConvTranspose2d(128, 40, kernel_size=3, padding=1, output_padding=1, stride=1),
			nn.BatchNorm2d(40),
			nn.RELU() )

		self.lstm = nn.LSTM(30*30, 30*30, 2, bidirectional=False, batch_first=True)


	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), x.size(1), -1)

		return self.lstm(x)
