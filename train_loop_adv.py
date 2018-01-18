import torch
from torch.autograd import Variable
import torch.nn.init as init

import numpy as np
import pickle

import os
from glob import glob
from tqdm import tqdm

class TrainLoop(object):

	def __init__(self, model, discriminator, optimizer, optimizer_d, train_loader, valid_loader, checkpoint_path=None, checkpoint_epoch=None, cuda=True):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.cuda_mode = cuda
		self.model = model
		self.discriminator = discriminator
		self.optimizer = optimizer
		self.optimizer_d = optimizer_d
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.history = {'train_loss': [], 'valid_loss': []}
		self.total_iters = 0
		self.cur_epoch = 0
		self.its_without_improv = 0
		self.last_best_val_loss = float('inf')

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))
		else:
			self.initialize_params()

	def train(self, n_epochs=1, patience = 5):

		while self.cur_epoch < n_epochs and self.its_without_improv < patience:
			print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
			train_iter = tqdm(enumerate(self.train_loader))

			train_loss = 0.0
			valid_loss = 0.0

			# Train step

			for t,batch in train_iter:
				new_train_loss = self.train_step(batch)
				train_loss += new_train_loss

			self.history['train_loss'].append(train_loss/(t+1))
			self.total_iters += 1

			# Validation

			for t, batch in enumerate(self.valid_loader):
				new_valid_loss = self.valid(batch)
				valid_loss += new_valid_loss

			self.history['valid_loss'].append(valid_loss/(t+1))

			print('Total train loss: {}'.format(self.history['train_loss'][-1]))
			print('Total valid loss: {}'.format(self.history['valid_loss'][-1]))

			self.cur_epoch += 1

			self.checkpointing()

			if valid_loss < self.last_best_val_loss:
				self.its_without_improv = 0
				self.last_best_val_loss = valid_loss
			else:
				self.its_without_improv += 1

			if self.its_without_improv > patience:
				self.update_lr()

		# saving final models
		print('Saving final model...')
		self.checkpointing()

	def train_step(self, batch):

		self.model.train()
		self.discriminator.train()
		self.optimizer.zero_grad()
		self.optimizer_d.zero_grad()

		x, y = batch

		y = y.view(y.size(0), y.size(3), y.size(1)*y.size(2))

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()

		x = Variable(x)
		y = Variable(y, requires_grad=False)

		out = self.model.forward(x)

		out_d = out.clone.detach()

		d_real = self.discriminator.forward(y)
		d_fake = self.discriminator.forward(out_d)

		loss_d = torch.nn.functional.binary_cross_entropy(d_real, torch.ones_like(d_real)) + torch.nn.functional.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
		loss_d.backward()
		self.optimizer_d.step()

		d_fake_ = self.discriminator.forward(out)

		loss = torch.nn.functional.mse_loss(out, y) + torch.nn.functional.binary_cross_entropy(d_fake_, torch.ones_like(d_fake_))

		loss.backward()
		self.optimizer.step()

		return loss.data[0]

	def valid(self, batch):

		self.model.eval()
		self.discriminator.eval()

		x, y = batch

		y = y.view(y.size(0), y.size(3), y.size(1)*y.size(2))

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()

		x = Variable(x)
		y = Variable(y, requires_grad=False)

		out = self.model.forward(x)


		d_fake_ = self.discriminator.forward(out)

		loss = torch.nn.functional.mse_loss(out, y) + torch.nn.functional.binary_cross_entropy(d_fake_, torch.ones_like(d_fake_))

		return loss.data[0]

	def checkpointing(self):

		# Checkpointing
		print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'disc_state': self.discriminator.state_dict(),
		'optimizer_state': self.optimizer.state_dict(),
		'optimizer_d_state': self.optimizer_d.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'cur_epoch': self.cur_epoch,
		'its_without_improve': self.its_without_improv,
		'last_best_val_loss': self.last_best_val_loss}
		torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt)
			# Load model state
			self.model.load_state_dict(ckpt['model_state'])
			self.discriminator.load_state_dict(ckpt['disc_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			self.optimizer_d.load_state_dict(ckpt['optimizer_d_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			self.its_without_improv = ckpt['its_without_improve']
			self.last_best_val_loss = ckpt['last_best_val_loss']

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def print_params_norms(self):
		norm = 0.0
		for params in list(self.model.parameters()):
			norm+=params.norm(2).data[0]
		print('Sum of weights norms: {}'.format(norm))


	def print_grad_norms(self):
		norm = 0.0
		for params in list(self.model.parameters()):
			norm+=params.grad.norm(2).data[0]
		print('Sum of grads norms: {}'.format(norm))

	def initialize_params(self):
		for layer in self.model.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal(layer.weight.data)
			elif isinstance(layer, torch.nn.BatchNorm2d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

		for layer in self.discriminator.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal(layer.weight.data)
			elif isinstance(layer, torch.nn.BatchNorm2d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def update_lr(self):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = max(param_group['lr']/10., 0.000001)
		print('updating lr of generator to: {}'.format(param_group['lr']))

		for param_group in self.optimizer_d.param_groups:
			param_group['lr'] = max(param_group['lr']/10., 0.000001)
		print('updating lr of discriminator to: {}'.format(param_group['lr']))
