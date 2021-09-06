import torch
import torchvision
import torch.nn.init as init

import numpy as np
import pickle

import os
from glob import glob
from tqdm import tqdm

class TrainLoop(object):

	def __init__(self, model, generator, optimizer, scheduler, train_loader, valid_loader, max_gnorm=10.0, checkpoint_path=None, checkpoint_epoch=None, cuda=True, logger=None):
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
		self.generator = generator
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.max_gnorm = max_gnorm
		self.history = {'train_loss': [], 'valid_loss': []}
		self.total_iters = 0
		self.cur_epoch = 0
		self.its_without_improv = 0
		self.last_best_val_loss = np.inf
		self.logger = logger

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))
		else:
			self.initialize_params()

	def train(self, n_epochs=1, save_every=10):

		while self.cur_epoch < n_epochs:
			print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
			train_iter = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

			train_loss = 0.0
			valid_loss = 0.0

			# Train step

			for t,batch in train_iter:
				new_train_loss = self.train_step(batch)
				train_loss += new_train_loss
				if self.logger:
					self.logger.add_scalar('Train/Train Loss', new_train_loss, self.total_iters)
				self.total_iters += 1

			self.history['train_loss'].append(train_loss/(t+1))

			# Validation

			for t, batch in enumerate(self.valid_loader):
				new_valid_loss, input_streaking_images, frames_list, target_scenes = self.valid(batch)
				valid_loss += new_valid_loss

			self.history['valid_loss'].append(valid_loss/(t+1))

			if self.logger:
				self.logger.add_scalar('Valid/MSE', self.history['valid_loss'][-1], self.total_iters)
				self.logger.add_scalar('Valid/Best_MSE', np.min(self.history['valid_loss']), self.total_iters)
				grid_streaking = torchvision.utils.make_grid(input_streaking_images)
				self.logger.add_image('Inputs', grid_streaking, self.total_iters)
				z_ = torch.randn(8, 128).view(-1, 128, 1, 1).to(target_scenes.device)
				grid_generator = torchvision.utils.make_grid(self.generator(z_))
				self.logger.add_image('Random frames', grid_generator, self.total_iters)
				self.logger.add_video('Reconstructed', torch.cat(frames_list, 1), self.total_iters)
				self.logger.add_video('Target_scenes', target_scenes.permute(0,4,1,2,3), self.total_iters)

			print('Total train loss: {}'.format(self.history['train_loss'][-1]))
			print('Total valid loss: {}'.format(self.history['valid_loss'][-1]))

			self.cur_epoch += 1

			if valid_loss < self.last_best_val_loss:
				self.its_without_improv = 0
				self.last_best_val_loss = valid_loss
				self.checkpointing()
			else:
				self.its_without_improv += 1
				if self.cur_epoch % save_every == 0:
					self.checkpointing()

			self.scheduler.step()

		# saving final models
		print('Saving final model...')
		self.checkpointing()

	def train_step(self, batch):

		self.generator.eval()
		self.model.train()
		self.optimizer.zero_grad()

		x, y = batch

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()

		out = self.model.forward(x)

		loss = 0.0

		for i in range(out.size(1)):
			gen_frame = self.generator(out[:,i,:])
			loss += torch.nn.functional.mse_loss(gen_frame, y[...,i])

		loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gnorm)
		self.optimizer.step()

		if self.logger:
			self.logger.add_scalar('Info/Grad_norm', grad_norm, self.total_iters)
			self.logger.add_scalar('Info/LR', self.optimizer.param_groups[0]['lr'], self.total_iters)
			self.logger.add_scalar('Info/Epoch', self.cur_epoch, self.total_iters)

		return loss.item()

	def valid(self, batch):

		self.generator.eval()
		self.model.eval()

		x, y = batch

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()

		with torch.no_grad():

			out = self.model.forward(x)

			loss = 0
			frames_list = []

			for i in range(out.size(1)):
				gen_frame = self.generator(out[:,i,:])
				loss += torch.nn.functional.mse_loss(gen_frame, y[...,i])
				frames_list.append(gen_frame.unsqueeze(1))

		return loss.item(), x, frames_list, y

	def checkpointing(self):

		# Checkpointing
		print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'optimizer_state': self.optimizer.state_dict(),
		'scheduler_state': self.scheduler.state_dict(),
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
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load scheduler state
			self.scheduler.load_state_dict(ckpt['scheduler_state'])
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
			norm+=params.norm(2).item()
		print('Sum of weights norms: {}'.format(norm))


	def print_grad_norms(self):
		norm = 0.0
		for params in list(self.model.parameters()):
			norm+=params.grad.norm(2).item()
		print('Sum of grads norms: {}'.format(norm))

	def initialize_params(self):
		for layer in self.model.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight.data)
			elif isinstance(layer, torch.nn.BatchNorm2d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()
