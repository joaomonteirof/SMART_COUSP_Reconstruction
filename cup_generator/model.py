import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(torch.nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		#linear layer
		self.linear = torch.nn.Sequential()

		linear = nn.Linear(100, 1024)

		self.linear.add_module('linear', linear)

		# Initializer
		nn.init.normal_(linear.weight, mean=0.0, std=0.02)
		nn.init.constant_(linear.bias, 0.0)

		# Batch normalization
		bn_name = 'bn0'
		self.linear.add_module(bn_name, torch.nn.BatchNorm1d(1024))

		# Activation
		act_name = 'act0'
		self.linear.add_module(act_name, torch.nn.ReLU())

		# Hidden layers
		num_filters = [1024, 1024, 512, 256, 256]
		self.hidden_layer = torch.nn.Sequential()
		for i in range(len(num_filters)):
			# Deconvolutional layer
			if i == 0:
				deconv = nn.ConvTranspose2d(1024, num_filters[i], kernel_size=5, stride=2, padding=0)
			elif i == 2 or i == 4:
				deconv = nn.ConvTranspose2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=1)
			else:
				deconv = nn.ConvTranspose2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=0)

			deconv_name = 'deconv' + str(i + 1)
			self.hidden_layer.add_module(deconv_name, deconv)

			# Initializer
			nn.init.normal_(deconv.weight, mean=0.0, std=0.02)
			nn.init.constant_(deconv.bias, 0.0)

			# Batch normalization
			bn_name = 'bn' + str(i + 1)
			self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

			# Activation
			act_name = 'act' + str(i + 1)
			self.hidden_layer.add_module(act_name, torch.nn.ReLU())

		# Output layer
		self.output_layer = torch.nn.Sequential()
		# Deconvolutional layer
		out = torch.nn.ConvTranspose2d(256, 1, kernel_size=4, stride=2, padding=1)
		self.output_layer.add_module('out', out)
		# Initializer
		nn.init.normal_(out.weight, mean=0.0, std=0.02)
		nn.init.constant_(out.bias, 0.0)
		# Activation
		self.output_layer.add_module('act', torch.nn.Sigmoid())

	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = self.linear(x)
		h = self.hidden_layer(x.view(x.size(0), 1024, 1, 1))
		out = self.output_layer(h)
		return out

class Discriminator(torch.nn.Module):
	def __init__(self, optimizer, lr, betas, batch_norm=False):
		super(Discriminator, self).__init__()

		self.projection = nn.Conv2d(1, 1, kernel_size=8, stride=2, padding=3, bias=False)
		with torch.no_grad():
			self.projection.weight.div_(torch.norm(self.projection.weight, keepdim=True))

		# Hidden layers
		self.hidden_layer = torch.nn.Sequential()
		num_filters = [256, 512, 512, 1024]
		for i in range(len(num_filters)):
			# Convolutional layer
			if i == 0:
				conv = nn.Conv2d(1, num_filters[i], kernel_size=4, stride=2, padding=1)
			elif i == 3:
				conv = nn.Conv2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=3, padding=1)
			else:
				conv = nn.Conv2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=1)

			conv_name = 'conv' + str(i + 1)
			self.hidden_layer.add_module(conv_name, conv)

			# Initializer
			nn.init.normal_(conv.weight, mean=0.0, std=0.02)
			nn.init.constant_(conv.bias, 0.0)

			if i != 0 and batch_norm:
			# Batch normalization
				bn_name = 'bn' + str(i + 1)
				self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

			# Activation
			act_name = 'act' + str(i + 1)
			self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

		# Output layer
		self.output_layer = torch.nn.Sequential()
		# Convolutional layer
		out = nn.Conv2d(num_filters[i], 1, kernel_size=4, stride=3, padding=1)
		self.output_layer.add_module('out', out)
		# Initializer
		nn.init.normal_(out.weight, mean=0.0, std=0.02)
		nn.init.constant_(out.bias, 0.0)
		# Activation
		self.output_layer.add_module('act', nn.Sigmoid())

		self.optimizer = optimizer(list(self.hidden_layer.parameters()) + list(self.output_layer.parameters()), lr=lr, betas=betas)

	def forward(self, x):
		p_x = self.projection(x)
		h = self.hidden_layer(p_x)
		out = self.output_layer(h)
		return out
