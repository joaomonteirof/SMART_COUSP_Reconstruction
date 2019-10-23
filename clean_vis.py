from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
import h5py
import torch
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms


class Loader(Dataset):

	def __init__(self, hdf5_name):
		super(Loader, self).__init__()
		self.hdf5_name = hdf5_name

		open_file = h5py.File(self.hdf5_name, 'r')
		self.length = len(open_file['data'])
		open_file.close()

		self.open_file = None

	def __getitem__(self, index):

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		scene = torch.from_numpy(np.moveaxis(self.open_file['data'][index], -1, 0)).float()

		idx = np.random.randint(scene.size(0))

		return scene.unsqueeze(0)

	def __len__(self):
		return self.length


def plot_real(n_tests, data_path):

	real_loader = Loader(hdf5_name = data_path)

	n_cols, n_rows = (n_tests, 100)
	fig, axes = plt.subplots(n_cols, n_rows, figsize=(n_rows, n_cols))

	intra_mse = []

	for i in range(n_tests):

		img_idx = np.random.randint(len(real_loader))
		real_sample = real_loader[img_idx].squeeze()
	
		for ax, img in zip(axes[i, :].flatten(), real_sample):
			ax.axis('off')
			ax.set_adjustable('box-forced')

			ax.imshow(img, cmap="gray", aspect='equal')
		
		plt.subplots_adjust(wspace=0, hspace=0)
		
		
		to_pil = transforms.ToPILImage()
		enhance = True

		if enhance:
			frames = [ImageEnhance.Sharpness( to_pil(frame.unsqueeze(0)) ).enhance(10.0) for frame in real_sample]
		else:
			frames = [to_pil(frame.unsqueeze(0)) for frame in real_sample]

		frames[0].save('video_'+str(i)+'_real.gif', save_all=True, append_images=frames[1:])


	save_fn = 'real.pdf'
	plt.savefig(save_fn)

	plt.close()


def save_gif(data, file_name, enhance):

	data = data.view([100, 32, 32])

	to_pil = transforms.ToPILImage()

	if enhance:
		frames = [ImageEnhance.Sharpness( to_pil(frame.unsqueeze(0)) ).enhance(10.0) for frame in data]
	else:
		frames = [to_pil(frame.unsqueeze(0)) for frame in data]

	frames[0].save(file_name, save_all=True, append_images=frames[1:])


if __name__ == '__main__':

	data = './train.hdf' 
	plot_real(5, data)
