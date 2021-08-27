import torch
import glob
import numpy as np
from torch.utils.data import Dataset
from data_prep.input_data_gen import *
import torchvision
import torchvision.transforms as transforms

_TRAIN_FRAMES_FRACTION = 0.9

def prep_video(data, im_size):

	data = data.float()/255.0

	data = data.permute(0,3,1,2)

	data = torchvision.transforms.Grayscale(1)(data)

	data = torchvision.transforms.Resize(im_size)(data)

	data = transforms.CenterCrop(im_size)(data)

	return data

def augment_video(vid_tensor):

	vid_tensor = augument_image(vid_tensor)
	if torch.rand(1).item() > 0.5:
		vid_tensor = torch.flip(vid_tensor, (0,))

	return vid_tensor

def augument_image(im_tensor):

	im_tensor = torchvision.transforms.RandomHorizontalFlip(p=0.5)(im_tensor)
	im_tensor = torchvision.transforms.RandomRotation(10)(im_tensor)
	im_tensor = torchvision.transforms.RandomVerticalFlip(p=0.2)(im_tensor)

	return im_tensor


class Loader(Dataset):

	def __init__(self, im_size, n_frames, data_path, mode, sample_size=1, mask_path=None):
		super(Loader, self).__init__()

		assert mode == "train" or mode == "test", "Mode should be set to train or test."

		self.mode = mode
		self.im_size = im_size
		self.n_frames = n_frames
		self.sample_size = sample_size

		files = glob.glob(f"{data_path}*.avi")

		self.data = []

		for f in files:
			video_data = torchvision.io.read_video(f)[0]
			video_data = prep_video(video_data, self.im_size)
			self.data.append(video_data)

		if mask_path:
			self.mask = torch.from_numpy(np.load(mask_path))
		else:
			self.mask = None

	def __getitem__(self, index):

		if index >= len(self):
			raise IndexError

		random_video_idx = torch.randint(0, len(self.data), (1,)).item()
		video = self.data[random_video_idx]

		split_idx = int(video.size(0)*_TRAIN_FRAMES_FRACTION)

		if self.mode == "train":
			start_idx = 0
			end_idx = split_idx
		else:
			start_idx = split_idx
			end_idx = video.size(0)

		random_start_idx = torch.randint(start_idx, end_idx-self.n_frames, (1,)).item()

		video = video[random_start_idx:(random_start_idx+self.n_frames),...]

		if self.mode == "train":
			video = augment_video(video)

		video = video.squeeze(1).permute(1, 2, 0)

		streaking_image = get_streaking_image(video.numpy(), mask=self.mask)

		streaking_image = torch.from_numpy(streaking_image).unsqueeze(0).float().contiguous()
		video = video.unsqueeze(0).float().contiguous()

		return streaking_image, video

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

		return x

	def __len__(self):
		return self.sample_size

if __name__ == "__main__":

	"""
	dataset = Loader_gen(256, "~/Downloads/papers_video/", sample_size=10)

	for i, el in enumerate(dataset):
		print(f"{i}/{len(dataset)}", el.size())
	"""

	dataset = Loader(256, 30, "~/Downloads/papers_video/", "train", sample_size=10, mask_path="./mask.npy")

	for i, el in enumerate(dataset):
		print(f"{i}/{len(dataset)}", el[0].size(), el[1].size())

