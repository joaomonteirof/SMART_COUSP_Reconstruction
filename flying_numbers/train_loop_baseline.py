import torch
import torch.nn.functional as F
import torchvision
import torch.nn.init as init
from pytorch_msssim import ms_ssim

import numpy as np
import pickle

import os
from glob import glob
from tqdm import tqdm

class TrainLoop(object):

	def __init__(self, model, optimizer, train_loader, valid_loader, max_gnorm=10.0, patience=10, lr_factor=0.1, checkpoint_path=None, checkpoint_epoch=None, cuda=True, logger=None):
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
		self.optimizer = optimizer
		self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=lr_factor, patience=patience, min_lr=1e-8)
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.max_gnorm = max_gnorm
		self.history = {'train_loss': [], 'valid_mse': [], 'valid_mssim': []}
		self.total_iters = 0
		self.cur_epoch = 0
		self.its_without_improv = 0
		self.last_best_val_mse = np.inf
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
			valid_mse = 0.0
			valid_mssim = 0.0

			# Train step

			for t,batch in train_iter:
				new_train_loss, new_mse, new_mssim = self.train_step(batch)
				train_loss += new_train_loss
				train_mse += new_train_mse
				train_mssim += new_train_mssim
				if self.logger:
					self.logger.add_scalar('Train/Train Loss', new_train_loss, self.total_iters)
					self.logger.add_scalar('Train/Train MSE', new_mse, self.total_iters)
					self.logger.add_scalar('Train/Train MS-SSIM', new_mssim, self.total_iters)
				self.total_iters += 1

			self.history['train_loss'].append(train_loss/(t+1))
			self.history['train_mse'].append(train_mse/(t+1))
			self.history['train_msssim'].append(train_mssim/(t+1))

			# Validation

			for t, batch in enumerate(self.valid_loader):
				new_valid_mse, new_valid_mssim, approximate_scenes, frames_list, target_scenes = self.valid(batch)
				valid_mse += new_valid_mse
				valid_mssim += new_valid_mssim

			self.history['valid_mse'].append(valid_mse/(t+1))
			self.history['valid_mssim'].append(valid_mssim/(t+1))

			if self.logger:
				self.logger.add_scalar('Info/Epoch', self.cur_epoch, self.total_iters)
				self.logger.add_scalar('Valid/MSE', self.history['valid_mse'][-1], self.total_iters)
				self.logger.add_scalar('Valid/Best_MSE', np.min(self.history['valid_mse']), self.total_iters)
				self.logger.add_scalar('Valid/MS-SSIM', self.history['valid_mssim'][-1], self.total_iters)
				self.logger.add_scalar('Valid/Best_MS-SSIM', np.max(self.history['valid_mssim']), self.total_iters)
				self.logger.add_video('Reconstructed', torch.cat(frames_list, 1), self.total_iters)
				self.logger.add_video('Approximate_scenes', approximate_scenes.permute(0,4,1,2,3), self.total_iters)				
				self.logger.add_video('Target_scenes', target_scenes.permute(0,4,1,2,3), self.total_iters)

			print('Total train loss: {}'.format(self.history['train_loss'][-1]))
			print('Valid MSE: {}'.format(self.history['valid_mse'][-1]))
			print('Valid MS-SSIM: {}'.format(self.history['valid_mssim'][-1]))

			self.cur_epoch += 1
			self.lr_scheduler.step(self.history['valid_mse'][-1])

			if self.history['valid_mse'][-1] < self.last_best_val_mse:
				self.its_without_improv = 0
				self.last_best_val_mse = self.history['valid_mse'][-1]
				self.checkpointing()
			else:
				self.its_without_improv += 1
				if self.cur_epoch % save_every == 0:
					self.checkpointing()

		# saving final models
		print('Saving final model...')
		self.checkpointing()

	def train_step(self, batch):

		self.model.train()
		self.optimizer.zero_grad()

		x, y = batch

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()

		out = self.model.forward(x)

		mse = torch.nn.functional.mse_loss(out, y)

		mssim = 0

		for i in range(out.size(-1)):
			gen_frame = out[...,i]
			mssim += ms_ssim(F.upsample(gen_frame, scale_factor=3), F.upsample(y[...,i], scale_factor=3))

		mssim = 1.0-mssim/(i+1)
		loss = mse + 0.7*mssim

		loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gnorm)
		self.optimizer.step()

		if self.logger:
			self.logger.add_scalar('Info/Grad_norm', grad_norm, self.total_iters)
			self.logger.add_scalar('Info/LR', self.optimizer.param_groups[0]['lr'], self.total_iters)

		return loss.item(), mse.item(), mssim.item()

	def valid(self, batch):

		self.model.eval()

		x, y = batch

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()

		with torch.no_grad():

			out = self.model.forward(x)

			mse = 0
			mssim = 0
			frames_list = []

			for i in range(out.size(-1)):
				gen_frame = out[...,i]
				mse += torch.nn.functional.mse_loss(gen_frame, y[...,i])
				mssim += ms_ssim(F.upsample(gen_frame, scale_factor=3), F.upsample(y[...,i], scale_factor=3))
				frames_list.append(gen_frame.unsqueeze(1))

		return mse.item(), (mssim/(i+1)).item(), x, frames_list, y

	def checkpointing(self):

		# Checkpointing
		print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'optimizer_state': self.optimizer.state_dict(),
		'scheduler_state': self.lr_scheduler.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'cur_epoch': self.cur_epoch,
		'its_without_improve': self.its_without_improv,
		'last_best_val_mse': self.last_best_val_mse}
		torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt)
			# Load model state
			self.model.load_state_dict(ckpt['model_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load Scheduler state
			self.lr_scheduler.load_state_dict(ckpt['scheduler_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			self.its_without_improv = ckpt['its_without_improve']
			self.last_best_val_mse = ckpt['last_best_val_mse']

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
