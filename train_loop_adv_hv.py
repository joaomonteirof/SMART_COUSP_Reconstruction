import torch
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np
import pickle

import os
from glob import glob
from tqdm import tqdm

class TrainLoop(object):

	def __init__(self, model, disc_list, optimizer, train_loader, valid_loader, nadir_slack=1.1, checkpoint_path=None, checkpoint_epoch=None, cuda=True):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt_model = os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.save_epoch_fmt_disc = os.path.join(self.checkpoint_path, 'D{}_checkpoint.pt')
		self.cuda_mode = cuda
		self.model = model
		self.disc_list = disc_list
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.history = {'hv': [], 'mse': [], 'adv': [], 'disc': [], 'valid_mse': []}
		self.total_iters = 0
		self.cur_epoch = 0
		self.its_without_improv = 0
		self.last_best_val_loss = np.inf
		self.nadir_slack = nadir_slack

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt_model.format(checkpoint_epoch))
		else:
			self.initialize_params()

	def train(self, n_epochs=1, patience = 5, save_every=10):

		while self.cur_epoch < n_epochs:
			print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
			train_iter = tqdm(enumerate(self.train_loader))

			hv_epoch=0.0
			mse_epoch=0.0
			adv_epoch=0.0
			disc_epoch=0.0
			valid_loss=0.0

			# Train step

			for t,batch in train_iter:
				hv, mse, adv, disc = self.train_step(batch)
				self.total_iters += 1
				hv_epoch+=hv
				mse_epoch+=mse
				adv_epoch+=adv
				disc_epoch+=disc

			self.history['hv'].append(hv_epoch/(t+1))
			self.history['mse'].append(mse_epoch/(t+1))
			self.history['adv'].append(adv_epoch/(t+1))
			self.history['disc'].append(disc_epoch/(t+1))

			# Validation

			for t, batch in enumerate(self.valid_loader):
				new_valid_loss = self.valid(batch)
				valid_loss += new_valid_loss

			self.history['valid_mse'].append(valid_loss/(t+1))

			print('NLH, MSE, Adversarial Loss, and Discriminators loss : {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}'.format(self.history['hv'][-1], self.history['mse'][-1], self.history['adv'][-1], self.history['disc'][-1]))
			print('Total valid MSE: {}'.format(self.history['valid_mse'][-1]))

			self.cur_epoch += 1

			if valid_loss < self.last_best_val_loss:
				self.its_without_improv = 0
				self.last_best_val_loss = valid_loss
				self.checkpointing()
			else:
				self.its_without_improv += 1
				if self.cur_epoch % save_every == 0:
					self.checkpointing()

			if self.its_without_improv > patience:
				#self.update_lr()
				self.its_without_improv = 0

		# saving final models
		print('Saving final model...')
		self.checkpointing()

	def train_step(self, batch):

		self.model.train()

		x, y = batch
		y_real_ = torch.ones(x.size(0))
		y_fake_ = torch.zeros(x.size(0))

		y = y.view(y.size(0), y.size(1), -1)

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()
			y_real_ = y_real_.cuda()
			y_fake_ = y_fake_.cuda()

		x = Variable(x)
		y = Variable(y, requires_grad=False)
		y_real_ = Variable(y_real_)
		y_fake_ = Variable(y_fake_)

		out = self.model.forward(x)

		out_d = out.detach()

		# train discriminators

		loss_d = 0

		for disc in self.disc_list:
			disc.optimizer.zero_grad()
			d_real = disc.forward(y).squeeze()
			d_fake = disc.forward(out_d).squeeze()
			loss_disc = F.binary_cross_entropy(d_real, y_real_) + F.binary_cross_entropy(d_fake, y_fake_)
			loss_disc.backward()
			disc.optimizer.step()
			loss_d += loss_disc.data[0]

		# train main model

		loss_model = 0
		loss_adv = 0
		losses_list_float = []
		losses_list_var = []

		for disc in self.disc_list:
			losses_list_var.append(F.binary_cross_entropy(disc.forward(out).squeeze(), y_real_))
			losses_list_float.append(losses_list_var[-1].data[0])

		rec_loss = F.mse_loss(out, y)
		losses_list_float.append(rec_loss.data[0])

		self.update_nadir_point(losses_list_float)

		for i, loss in enumerate(losses_list_var):
			loss_model -= torch.log(self.nadir - loss)
			loss_adv += loss.data[0]

		loss_model -= torch.log(self.nadir - rec_loss)

		self.optimizer.zero_grad()
		loss_model.backward()
		self.optimizer.step()

		return loss_model.data[0], rec_loss.data[0], loss_adv, loss_d

	def valid(self, batch):

		self.model.eval()

		x, y = batch

		y = y.view(y.size(0), y.size(1), -1)

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()

		x = Variable(x)
		y = Variable(y, requires_grad=False)

		out = self.model.forward(x)

		loss = F.mse_loss(out, y)

		return loss.data[0]

	def checkpointing(self):

		# Checkpointing
		print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'optimizer_state': self.optimizer.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'cur_epoch': self.cur_epoch,
		'its_without_improve': self.its_without_improv,
		'last_best_val_loss': self.last_best_val_loss}
		torch.save(ckpt, self.save_epoch_fmt_model.format(self.cur_epoch))

		for i, disc in enumerate(self.disc_list):
			ckpt = {'model_state': disc.state_dict(),
				'optimizer_state': disc.optimizer.state_dict()}
			torch.save(ckpt, self.save_epoch_fmt_disc.format(i + 1))

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt)
			# Load model state
			self.model.load_state_dict(ckpt['model_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			self.its_without_improv = ckpt['its_without_improve']
			self.last_best_val_loss = ckpt['last_best_val_loss']

			for i, disc in enumerate(self.disc_list):
				ckpt = torch.load(self.save_epoch_fmt_disc.format(i + 1))
				disc.load_state_dict(ckpt['model_state'])
				disc.optimizer.load_state_dict(ckpt['optimizer_state'])

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def update_nadir_point(self, losses_list):
		self.nadir = float(np.max(losses_list) * self.nadir_slack + 1e-8)

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
