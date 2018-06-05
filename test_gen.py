from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
import models_zoo
from cup_generator.model import Generator
from data_load import Loader
from train_loop import TrainLoop

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageEnhance

import torchvision.transforms as transforms

def test_model(model, data_loader, n_tests, cuda_mode, enhancement):

	model.eval()

	to_pil = transforms.ToPILImage()

	for i in range(n_tests):
		img_idx = np.random.randint(len(data_loader))
		sample_in, sample_out = data_loader[img_idx]

		sample_in = sample_in.view(1, sample_in.size(0), sample_in.size(1), sample_in.size(2))

		if cuda_mode:
			sample_in = sample_in.cuda()

		sample_in = Variable(sample_in)

		out = model.forward(sample_in)

		frames_list = []

		for i in range(out.size(1)):
			gen_frame = self.generator(out[:,i,:].squeeze().contiguous())
			frames_list.append(gen_frame.cpu().data.squeeze(0))

		sample_rec = torch.cat(frames_list, 0)

		save_gif(sample_out, str(i+1)+'_real.gif', enhance=False)
		save_gif(sample_rec, str(i+1)+'_rec.gif', enhance=enhancement)

def save_gif(data, file_name, enhance):

	data = data.view([40, 30, 30])

	to_pil = transforms.ToPILImage()

	if enhance:
		frames = [ImageEnhance.Sharpness( to_pil(frame.unsqueeze(0)) ).enhance(10.0) for frame in data]
	else:
		frames = [to_pil(frame.unsqueeze(0)) for frame in data]

	frames[0].save(file_name, save_all=True, append_images=frames[1:])


def plot_learningcurves(history, keys):

	for i, key in enumerate(keys):
		plt.figure(i+1)
		plt.plot(history[key])
		plt.title(key)
	
	plt.show()

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing online transfer learning for emotion recognition tasks')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--generator-path', type=str, default=None, metavar='Path', help='Path for generator params')
	parser.add_argument('--input-data-path', type=str, default='./data/input/', metavar='Path', help='Path to data input data')
	parser.add_argument('--targets-data-path', type=str, default='./data/targets/', metavar='Path', help='Path to output data')
	parser.add_argument('--n-tests', type=int, default=4, metavar='N', help='number of samples to  (default: 64)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--no-plots', action='store_true', default=False, help='Disables plot of train/test losses')
	parser.add_argument('--enhance', action='store_true', default=True, help='Enables enhancement')
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	data_set = Loader(input_file_name=args.input_data_path+'input_train_3.hdf', output_file_name=args.targets_data_path+'output_train_3.hdf')
	#data_set = Loader(input_file=args.input_data_path+'input_valid.hdf', output_file=args.targets_data_path+'output_valid.hdf')

	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)

	model = models_zoo.model_3d_lstm_gen(args.cuda)
	generator = Generator().eval()

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)

	history = ckpt['history']

	model.load_state_dict(ckpt['model_state'])

	gen_state = torch.load(args.generator_path, map_location=lambda storage, loc: storage)
	generator.load_state_dict(gen_state['model_state'])

	if args.cuda:
		model = model.cuda()
		generator = generator.cuda()

	test_model(model=model, generator=generator, data_loader=data_set, n_tests=args.n_tests, cuda_mode=args.cuda, enhancement=args.enhance)

	if not args.no_plots:
		plot_learningcurves(history, list(history.keys()))
