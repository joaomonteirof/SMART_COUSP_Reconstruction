import argparse
import pickle
import h5py
import numpy as np
import scipy.io as sio
import matplotlib
import random
import cv2
import torch

MATRIX_TRANSFORM = np.array([[1.0079, 0.0085, 0.0001],
						[0.0226, 1.0155, 0.0001],
						[0.9163, 0.6183, 1.0000]]).T


def normalize(data):
	data_max, data_min = np.max(data), np.min(data)
	return (data-data_min) / (data_max - data_min + 1e-8)

def to_binary(img, level):

	bin_img = np.zeros(img.shape)

	idx = np.where(img>level)

	bin_img[idx]=1.

	return bin_img

def get_streaking_image(x, mask=None, intensity_variation=True):

	start_idx =random.randint(0, x.shape[-1]//2-1)
	x = x[..., start_idx:(start_idx+10)]

	D_x, D_y, D_t = x.shape

	if mask is None:
		mask = np.ones([D_x, D_y])

	x[:, :, :3] = 0.0 ## ignores first 3 frames

	Cu=np.zeros([D_x,D_y+D_t-1,D_t])

	for i in range(D_t):
		if i>=3:
			Cu[:,i:i+D_y,i]=mask

	x_out = np.zeros(Cu.shape)

	for i in range(D_t):
		idx = D_t-i-1
		im=x[:,:,idx].T
		if idx>=3:
			if intensity_variation:
				im *= (1.0-0.1*random.random()) ## randomly changes the intensity of each frame to simulate the fluctuation of laser intensity
			im = cv2.warpPerspective(src=im, M=MATRIX_TRANSFORM, dsize=(D_x, D_y)).T
		x_out[:,i:i+D_y,i] = im

	y1=np.multiply(x_out, Cu)
	y1 = y1.sum(2)
	y1 = normalize(y1)

	return y1

if __name__ == "__main__":

	# Data settings
	parser = argparse.ArgumentParser(description='Generate Bouncing balls dataset')
	parser.add_argument('--data-path', type=str, default='./', metavar='Path', help='Path for input data')
	args = parser.parse_args()

	data = torch.load(args.data_path)

	streaking_image = get_streaking_image(data[10].numpy())

	print(streaking_image.shape)

	import matplotlib.pyplot as plt

	plt.imshow(streaking_image)
	plt.show()