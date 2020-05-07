from __future__ import print_function
import argparse
import torch
import models_zoo
from cup_generator.model import Generator
from data_load import Loader
from train_loop import TrainLoop
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
from data_prep.offline_input_data_gen import *
import torchvision.transforms as transforms

def test_model(model, generator, data, cuda_mode, out_path):

	model.eval()
	to_pil = transforms.ToPILImage()

	with torch.no_grad():

		for i in range(data.size(0)):
			test_example = data[i:i+1]
			to_pil((test_example[0]*255).to(torch.uint8)).save(str(i+1)+'_streaking.png')

			if cuda_mode:
				test_example = test_example.cuda()

			out = model.forward(test_example)

			frames_list = []

			for j in range(out.size(1)):

				gen_frame = generator(out[:,j,:].contiguous())
				frames_list.append(gen_frame.cpu().squeeze(0).detach())

			sample_rec = torch.cat(frames_list, 0)

			save_gif(sample_rec, out_path+str(i+1)+'_rec.gif')

def save_gif(data, file_name):

	to_pil = transforms.ToPILImage()

	frames = [to_pil(frame.unsqueeze(0)) for frame in data]

	frames[0].save(file_name, save_all=True, append_images=frames[1:])


if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Generate reconstructed samples from exp data')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--generator-path', type=str, default=None, metavar='Path', help='Path for generator params')
	parser.add_argument('--data-path', type=str, default=None, metavar='Path', help='Path to input data - streaking images in different .mat file each')
	parser.add_argument('--video-path', type=str, default=None, metavar='Path', help='Path to input data - videos packed into single .mat')
	parser.add_argument('--mask-path', type=str, default=None, metavar='Path', help='Only used for the video path option')
	parser.add_argument('--n-frames', type=int, default=100, metavar='N', help='Number of frames per sample (default: 100)')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output results to')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	assert args.data_path is not None or args.video_path is not None, 'Set one of --data-path or --video-path'

	if args.data_path:
		data = glob.glob(args.data_path + '*.mat')

		if len(data)>0:

			im_list = []

			for im in data:
				im = sio.loadmat(im)
				im = im[sorted(im.keys())[-1]]
				im = torch.from_numpy(im).float().unsqueeze(0).unsqueeze(0)
				if im.max()>1:
					im/=255.
				im_list.append(im)

			im_data = torch.cat(im_list, 0).float()

		else:
			data = glob.glob(args.data_path + '*.png')
			im_list = []

			for im in data:
				im = Image.open(im)
				im_list.append(transforms.ToTensor()(im).unsqueeze(0))

			im_data = torch.cat(im_list, 0).float()

	else:

		data = sio.loadmat(args.video_path)['Data']

		print(data.shape)

		if args.mask_path:
			mask = sio.loadmat(args.mask_path)
			mask = mask[sorted(mask.keys())[-1]]
		else:
			mask = None

		im_list = []

		for i in range(data.shape[0]):
			im = get_streaking_image(data[i], mask)
			im_list.append(torch.from_numpy(im).unsqueeze(0).unsqueeze(0))

		im_data = torch.cat(im_list, 0).float()

	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)

	model = models_zoo.model_gen(n_frames=args.n_frames, cuda_mode=args.cuda)
	generator = Generator().eval()

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	model.load_state_dict(ckpt['model_state'], strict=True)

	gen_state = torch.load(args.generator_path, map_location=lambda storage, loc: storage)
	generator.load_state_dict(gen_state['model_state'])

	if args.cuda:
		model = model.cuda()
		generator = generator.cuda()

	test_model(model=model, generator=generator, data=im_data, cuda_mode=args.cuda, out_path=args.out_path)
