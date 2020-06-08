import h5py
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from data_prep.input_data_gen import *
import torchvision
import torchvision.transforms as transforms

class Loader(Dataset):

	def __init__(self, im_size, n_objects, n_frames, rep_times, sample_size, mask_path=None):
		super(Loader, self).__init__()
		self.im_size = im_size
		self.n_objects = n_objects
		self.n_frames = n_frames
		self.rep_times = rep_times
		self.sample_size = sample_size
		self.digit_size_ = 21
		self.step_length_ = 0.1
		self.mnist = torchvision.datasets.MNIST('./', train=True, transform=transforms.Compose([transforms.CenterCrop(21), transforms.ToTensor()]), target_transform=None, download=True)

		if mask_path:
			self.mask = np.load(mask_path)
		else:
			self.mask = None

	def __getitem__(self, index):

		inp, out = self.gen_example()

		return inp, out

	def __len__(self):
		return self.sample_size

	def gen_example(self):

		out = self.generate_moving_mnist()
		out = np.repeat(out, self.rep_times, axis=-1)
		inp = get_streaking_image(out, self.mask)

		out = torch.from_numpy(out).unsqueeze(0).float().contiguous()
		inp = torch.from_numpy(inp).unsqueeze(0).float().contiguous()

		return inp, out

	def generate_moving_mnist(self):
		'''
		Get random trajectories for the digits and generate a video.
		'''
		data = np.zeros((self.n_frames, self.im_size, self.im_size), dtype=np.float32)
		for n in range(self.n_objects):
			# Trajectory
			start_y, start_x = self.get_random_trajectory()
			ind = random.randint(0, len(self.mnist) - 1)
			digit_image = self.mnist[ind][0].numpy()
			for i in range(self.n_frames):
				top	= start_y[i]
				left   = start_x[i]
				bottom = top + self.digit_size_
				right  = left + self.digit_size_
				# Draw digit
				data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

		data = data.transpose(1,2,0)

		return data

	def get_random_trajectory(self):
		''' Generate a random sequence of a MNIST digit '''
		canvas_size = self.im_size - self.digit_size_
		x = random.random()
		y = random.random()
		theta = random.random() * 2 * np.pi
		v_y = np.sin(theta)
		v_x = np.cos(theta)

		start_y = np.zeros(self.n_frames)
		start_x = np.zeros(self.n_frames)
		for i in range(self.n_frames):
			# Take a step along velocity.
			y += v_y * self.step_length_
			x += v_x * self.step_length_

			# Bounce off edges.
			if x <= 0:
				x = 0
				v_x = -v_x
			if x >= 1.0:
				x = 1.0
				v_x = -v_x
			if y <= 0:
				y = 0
				v_y = -v_y
			if y >= 1.0:
				y = 1.0
				v_y = -v_y
			start_y[i] = y
			start_x[i] = x

		# Scale to the size of the canvas.
		start_y = (canvas_size * start_y).astype(np.int32)
		start_x = (canvas_size * start_x).astype(np.int32)

		return start_y, start_x

class Loader_gen(Dataset):

	def __init__(self, im_size, n_objects, n_frames, sample_size):
		super(Loader_gen, self).__init__()
		self.im_size = im_size
		self.n_objects = n_objects
		self.n_frames = n_frames
		self.sample_size = sample_size
		self.digit_size_ = 21
		self.step_length_ = 0.1
		self.mnist = torchvision.datasets.MNIST('./', train=True, transform=transforms.Compose([transforms.CenterCrop(21), transforms.ToTensor()]), target_transform=None, download=True)

	def __getitem__(self, index):

		out = self.generate_moving_mnist()
		out = torch.from_numpy(out[:, :, random.randint(0, self.n_frames-1)]).squeeze().unsqueeze(0).float().contiguous()

		return out

	def __len__(self):
		return self.sample_size

	def generate_moving_mnist(self):
		'''
		Get random trajectories for the digits and generate a video.
		'''
		data = np.zeros((self.n_frames, self.im_size, self.im_size), dtype=np.float32)
		for n in range(self.n_objects):
			# Trajectory
			start_y, start_x = self.get_random_trajectory()
			ind = random.randint(0, len(self.mnist) - 1)
			digit_image = self.mnist[ind][0].numpy()
			for i in range(self.n_frames):
				top	= start_y[i]
				left   = start_x[i]
				bottom = top + self.digit_size_
				right  = left + self.digit_size_
				# Draw digit
				data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

		data = data.transpose(1,2,0)

		return data

	def get_random_trajectory(self):
		''' Generate a random sequence of a MNIST digit '''
		canvas_size = self.im_size - self.digit_size_
		x = random.random()
		y = random.random()
		theta = random.random() * 2 * np.pi
		v_y = np.sin(theta)
		v_x = np.cos(theta)

		start_y = np.zeros(self.n_frames)
		start_x = np.zeros(self.n_frames)
		for i in range(self.n_frames):
			# Take a step along velocity.
			y += v_y * self.step_length_
			x += v_x * self.step_length_

			# Bounce off edges.
			if x <= 0:
				x = 0
				v_x = -v_x
			if x >= 1.0:
				x = 1.0
				v_x = -v_x
			if y <= 0:
				y = 0
				v_y = -v_y
			if y >= 1.0:
				y = 1.0
				v_y = -v_y
			start_y[i] = y
			start_x[i] = x

		# Scale to the size of the canvas.
		start_y = (canvas_size * start_y).astype(np.int32)
		start_x = (canvas_size * start_x).astype(np.int32)
		return start_y, start_x

if __name__=='__main__':

	test_dataset = Loader(im_size=64, n_objects=2, n_frames=40, rep_times=1, sample_size=100)

	print(test_dataset.mask)

	inp_, out_ = test_dataset.__getitem__(10)

	out_ = out_.squeeze(0).numpy()

	print(inp_.shape, out_.shape)

	import matplotlib.pyplot as plt

	im = plt.imshow(out_[...,-1])
	for i in range(out_.shape[-1]):
		im.set_data(out_[...,i])
		plt.pause(0.02)
	plt.show()

	print(inp_.max(), out_.max())

	print(inp_.min(), out_.min())

	test_dataset = Loader_gen(im_size=64, n_objects=2, n_frames=40, sample_size=100)

	out_ = test_dataset.__getitem__(10)

	print(out_.shape)
