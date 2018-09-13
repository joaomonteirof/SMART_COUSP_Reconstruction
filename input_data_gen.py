from numpy import *				
from scipy import *			   
import argparse
import pickle
import h5py
import numpy as np
import scipy.io as sio

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def to_binary(img, level):

	bin_img = np.zeros(img.shape)

	idx = np.where(img>level)

	bin_img[idx]=1.

	return bin_img

def get_streaking_image(x, mask):

	D_x, D_y, D_t = x.shape
	'''
	p=4

	C = mask

	C_transiton=np.zeros([p*C.shape[0], p*C.shape[1]])

	for i in range(C.shape[0]):
		for j in range(C.shape[1]):
			C_transiton[p*i:p*(i+1),p*j:p*(j+1)]=C[i,j]

	C=to_binary(C_transiton[:D_x, :D_y], 0.1)
	'''

	C = mask

	C_1=np.zeros([D_x+D_t-1, D_y, D_t])

	for i in range(D_t):
		C_1[i:i+D_x,:,i]=C
	
	Cu=C_1

	x_1=np.zeros([D_x+D_t-1, D_y, D_t])

	for i in range(D_t):
		x_1[i:i+D_x,:,i]=x[:,:,i]

	x=x_1

	y1=np.multiply(x,Cu)

	y = y1.sum(2).T

	return y
		
if __name__ == "__main__":

	# Data settings
	parser = argparse.ArgumentParser(description='Generate Bouncing balls dataset')
	parser.add_argument('--output-path', type=str, default='./', metavar='Path', help='Path for output')
	parser.add_argument('--input-file-name', type=str, default='input.hdf', metavar='Path', help='input file name')
	parser.add_argument('--output-file-name', type=str, default='output.hdf', metavar='Path', help='Output file name')
	parser.add_argument('--mask-path', type=str, default='./mask.mat', metavar='Path', help='Output file name')
	args = parser.parse_args()

	r_mask = sio.loadmat(args.mask_path)['mask2']

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
