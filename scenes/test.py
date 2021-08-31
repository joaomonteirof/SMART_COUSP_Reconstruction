from __future__ import print_function
import argparse
import torch
import models_zoo
from cup_generator.model import Generator
from data_load import Loader
from train_loop import TrainLoop

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageEnhance

import torchvision.transforms as transforms

def test_model(model, generator, dataset, cuda_mode, enhancement):

	model.eval()
	to_pil = transforms.ToPILImage()

	with torch.no_grad():

		sample_in, sample_out = dataset[0]

		if cuda_mode:
			sample_in = sample_in.cuda()

		out = model.forward(sample_in)

		rec_frames_list = []
		out_frames_list = []


		for i in range(out.size(0)):
			for j in range(out.size(1)):

				gen_frame = generator(out[i,j,:].unsqueeze(0))
				rec_frames_list.append(gen_frame.cpu().squeeze(0).detach())
				out_frames_list.append(sample_out[i,:,:,:,j])


		sample_rec = torch.cat(rec_frames_list, 0)
		sample_out = torch.cat(out_frames_list, 0)

		save_gif(sample_out, 'real.gif', enhance=False)
		save_gif(sample_rec, 'rec.gif', enhance=enhancement)

def save_gif(data, file_name, enhance):

	to_pil = transforms.ToPILImage()

	if enhance:
		frames = [ImageEnhance.Sharpness( to_pil(frame.unsqueeze(0)) ).enhance(10.0) for frame in data]
	else:
		frames = [to_pil(frame.unsqueeze(0)) for frame in data]

	frames[0].save(file_name, save_all=True, append_images=frames[1:])


if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Generate reconstructed samples')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--generator-path', type=str, default=None, metavar='Path', help='Path for generator params')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--enhance', action='store_true', default=False, help='Enables enhancement')
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
	### Data options
	parser.add_argument('--im-size', type=int, default=256, metavar='N', help='H and W of frames (default: 256)')
	parser.add_argument('--n-frames', type=int, default=30, metavar='N', help='Number of frames per sample (default: 30)')
	parser.add_argument('--train-examples', type=int, default=50000, metavar='N', help='Number of training examples (default: 50000)')
	parser.add_argument('--val-examples', type=int, default=5000, metavar='N', help='Number of validation examples (default: 500)')
	parser.add_argument('--mask-path', type=str, default="./mask.npy", metavar='Path', help='path to encoding mask')
	parser.add_argument('--data-path', type=str, default="./", metavar='Path', help='path to data')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False


	data_set = Loader(args.im_size, args.n_frames, args.data_path, "test_full", mask_path=args.mask_path)

	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)

	model = models_zoo.model_gen(n_frames=args.n_frames, cuda_mode=args.cuda)
	generator = Generator().eval()

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	print(model.load_state_dict(ckpt['model_state'], strict=True))

	gen_state = torch.load(args.generator_path, map_location=lambda storage, loc: storage)
	print(generator.load_state_dict(gen_state['model_state'], strict=True))

	if args.cuda:
		model = model.cuda()
		generator = generator.cuda()

	test_model(model=model, generator=generator, dataset=data_set, cuda_mode=args.cuda, enhancement=args.enhance)
