from numpy import *				
from scipy import *			   
import argparse
import pickle
import h5py
import numpy as np
import scipy.io as sio
from skimage.transform import ProjectiveTransform, warp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

MATRIX_TRANSFORM = np.array([[1.0079, 0.0085, 0.0001],
						[0.0226, 1.0155, 0.0001],
						[0.9163, 0.6183, 1.0000]])

TRANSFORM = ProjectiveTransform(MATRIX_TRANSFORM)

def normalize(data):
	data_max, data_min = np.max(data), np.min(data)
	return (data-data_min) / (data_max - data_min + 1e-8)

def to_binary(img, level):

	bin_img = np.zeros(img.shape)

	idx = np.where(img>level)

	bin_img[idx]=1.

	return bin_img

def get_streaking_image(x, mask=None):

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
		im=x[:,:,idx:idx+1] * (1.0-0.1*random.random()) ## randomly changes the intensity of each frame to simulate the fluctuation of laser intensity
		bar_T = warp(im, TRANSFORM).squeeze(-1)
		x_out[:,i:i+D_y,i] = bar_T

	y1=np.multiply(x_out, Cu)
	y1 = y1.sum(2)
	y1 = normalize(y1)

	return y1
		
if __name__ == "__main__":

	# Data settings
	parser = argparse.ArgumentParser(description='Generate Bouncing balls dataset')
	parser.add_argument('--output-path', type=str, default='./', metavar='Path', help='Path for output')
	parser.add_argument('--input-file-name', type=str, default='input.hdf', metavar='Path', help='input file name')
	parser.add_argument('--output-file-name', type=str, default='output.hdf', metavar='Path', help='Output file name')
	parser.add_argument('--mask-path', type=str, default=None, metavar='Path', help='Mask .mat file')
	args = parser.parse_args()

	if args.mask_path:
		r_mask = r_mask[sorted(r_keys.mask())[-1]]
		r_mask = sio.loadmat(args.mask_path)
	else:
		r_mask = None

	hdf_input = h5py.File(args.input_file_name, 'r')
	hdf = h5py.File(args.output_path+args.output_file_name, 'w')

	for i in range(hdf_input['data'].shape[0]):
		sample=get_streaking_image(hdf_input['data'][i], r_mask)
		dat = np.expand_dims( sample, axis=0 )

		try:
			hdf['data'].resize(hdf['data'].shape[0]+1, axis=0)
			hdf['data'][-1:]=dat

		except KeyError:
			hdf.create_dataset('data', data=dat, maxshape=(None, sample.shape[0], sample.shape[1]))

		print(i)

	print(hdf['data'].shape)
	hdf.close()
