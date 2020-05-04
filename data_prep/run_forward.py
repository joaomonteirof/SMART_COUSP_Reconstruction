from __future__ import print_function
import argparse
import os
import torch
import scipy.io as sio
import numpy as np
import glob
from PIL import Image
from offline_input_data_gen import *
import torchvision.transforms as transforms


if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Compute simulated streaking images from set of scenes')
	parser.add_argument('--video-path', type=str, default=None, required=True, metavar='Path', help='Path to input data - videos packed into single .mat')
	parser.add_argument('--mask-path', type=str, default=None, metavar='Path', help='Only used for the video path option')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output results to')
	args = parser.parse_args()

	to_pil = transforms.ToPILImage()

	data = sio.loadmat(args.video_path)['Data']
	print('\nScenes loaded from {}'.format(args.video_path))
	print('\n', data.shape)

	if args.mask_path:
		mask = sio.loadmat(args.mask_path)
		mask = mask[sorted(mask.keys())[-1]]
		print('\nMask loaded from {}'.format(args.mask_path))
		print('\n', mask.shape)
	else:
		mask = None

	for i in range(data.shape[0]):
		im_name = os.path.join(args.out_path, str(i+1)+'_streaking.png')
		im = get_streaking_image(data[i], mask)
		im = torch.from_numpy(im).unsqueeze(0).float()
		to_pil((im*255).to(torch.uint8)).save(im_name)
		print('Saved image {}: {}'.format(i+1, im_name))