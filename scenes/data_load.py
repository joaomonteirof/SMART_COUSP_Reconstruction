import torch
import glob
import numpy as np
from torch.utils.data import Dataset
from data_prep.input_data_gen import *
import torchvision
import torchvision.transforms as transforms

def prep_video(data, im_size):

	data = data.float()/255.0

	data = data.permute(0,3,1,2)

	data = torchvision.transforms.Grayscale(1)(data)

	data = torchvision.transforms.Resize(im_size)(data)

	data = transforms.CenterCrop(im_size)(data)

	return data

def augument_image(im_tensor):

	im_tensor = torchvision.transforms.RandomHorizontalFlip(p=0.5)(im_tensor)
	im_tensor = torchvision.transforms.RandomRotation(10)(im_tensor)
	im_tensor = torchvision.transforms.RandomVerticalFlip(p=0.2)(im_tensor)

	return im_tensor


class Loader(Dataset):

	def __init__(self, im_size, n_objects, n_frames, mask_path=None):
		super(Loader, self).__init__()
		self.im_size = im_size
		self.n_objects = n_objects
		self.n_frames = n_frames
		self.rep_times = rep_times

		if mask_path:
			self.mask = np.load(mask_path)
		else:
			self.mask = None

	def __getitem__(self, index):

		inp, out = self.gen_example()
		if self.baseline_mode:
			if self.mask is None:
				inp = get_video_from_streaking_image(inp, self.n_frames, np.ones(self.im_size, self.im_size))
			else:
				inp = get_video_from_streaking_image(inp, self.n_frames, self.mask)

		return inp, out

	def __len__(self):
		return self.sample_size


class Loader_gen(Dataset):

	def __init__(self, im_size, data_path, sample_size=1):
		super(Loader_gen, self).__init__()
		self.im_size = im_size
		self.sample_size = sample_size

		files = glob.glob(f"{data_path}*.avi")

		self.data = []

		for f in files:
			video_data = torchvision.io.read_video(f)[0]
			video_data = prep_video(video_data, self.im_size)
			self.data.append(video_data)


	def __getitem__(self, index):

		if index >= len(self):
			raise IndexError

		random_video_idx = torch.randint(0, len(self.data), (1,)).item()
		video = self.data[random_video_idx]
		random_frame_idx = torch.randint(0, video.size(0), (1,)).item()

		x = augument_image(video[random_frame_idx])

		return video[random_frame_idx]

	def __len__(self):
		return self.sample_size

if __name__ == "__main__":

	dataset = Loader_gen(256, "/Users/joaomonteirof/Downloads/papers_video/", sample_size=10)

	for i, el in enumerate(dataset):
		print(f"{i}/{len(dataset)}", el.size())

