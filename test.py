from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
import models_zoo
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

		sample_rec = model.forward(sample_in).cpu().data[0]

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


def plot_learningcurves(history, *keys):

	for key in keys:
		plt.plot(history[key])
	
	plt.show()

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing online transfer learning for emotion recognition tasks')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--input-data-path', type=str, default='./data/input/', metavar='Path', help='Path to data input data')
	parser.add_argument('--targets-data-path', type=str, default='./data/targets/', metavar='Path', help='Path to output data')
	parser.add_argument('--n-tests', type=int, default=4, metavar='N', help='number of samples to  (default: 64)')
	parser.add_argument('--ngpus', type=int, default=0, help='Number of GPUs to use. Default=0 (no GPU)')
	parser.add_argument('--no-plots', action='store_true', default=False, help='Disables plot of train/test losses')
	parser.add_argument('--enhance', action='store_true', default=True, help='Enables enhancement')
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
	args = parser.parse_args()
	args.cuda = True if args.ngpus>0 and torch.cuda.is_available() else False

	#data_set = Loader(input_file=args.input_data_path+'input_train.hdf', output_file=args.targets_data_path+'output_train.hdf')
	data_set = Loader(input_file=args.input_data_path+'input_valid.hdf', output_file=args.targets_data_path+'output_valid.hdf')

	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)

	#model = models_zoo.model(args.cuda)
	#model = models_zoo.small_model(args.cuda)
	#model = models_zoo.model_cnn3d(args.cuda)
	model = models_zoo.model_3d_lstm(args.cuda)

	if args.ngpus > 1:
		model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpus)))

	if args.cuda:
		model = model.cuda()

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)

	history = ckpt['history']

	if not args.no_plots:
		plot_learningcurves(history, 'train_loss')
		plot_learningcurves(history, 'valid_loss')

	model.load_state_dict(ckpt['model_state'])
	test_model(model=model, data_loader=data_set, n_tests=args.n_tests, cuda_mode=args.cuda, enhancement=args.enhance)
