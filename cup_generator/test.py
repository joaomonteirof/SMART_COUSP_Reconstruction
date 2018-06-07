from __future__ import print_function

import argparse
import os
import matplotlib.pyplot as plt
import torch.utils.data
import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable
from torchvision.transforms import transforms
from model import *
from PIL import ImageEnhance

def denorm(unorm):
	norm = (unorm + 1) / 2

	return norm.clamp(0, 1)


def plot_learningcurves(history, keys):

	for i, key in enumerate(keys):
		plt.figure(i+1)
		plt.plot(history[key])
		plt.title(key)
	
	plt.show()


def test_model(model, n_tests, cuda_mode, enhance=True):
	model.eval()

	to_pil = transforms.ToPILImage()
	to_tensor = transforms.ToTensor()

	z_ = torch.randn(n_tests, 100).view(-1, 100, 1, 1)

	if cuda_mode:
		z_ = z_.cuda()

	z_ = Variable(z_)
	out = model.forward(z_)

	for i in range(out.size(0)):
		#sample = denorm(out[i].data)
		sample = out[i].data

		if len(sample.size())<3:
			sample = sample.view(1, 28, 28)

		if enhance:
			sample = ImageEnhance.Sharpness( to_pil(sample.cpu()) ).enhance(1.2)
		else:
			sample = to_pil(sample.cpu())

		sample.save('sample_{}.pdf'.format(i + 1))

def save_samples(generator: torch.nn.Module, cp_name: str, cuda_mode: bool, prefix: str, save_dir='./', nc=3, im_size=64, fig_size=(5, 5), enhance=True):
	generator.eval()

	n_tests = fig_size[0] * fig_size[1]

	to_pil = transforms.ToPILImage()
	to_tensor = transforms.ToTensor()

	noise = torch.randn(n_tests, 100).view(-1, 100, 1, 1)

	if cuda_mode:
		noise = noise.cuda()

	noise = Variable(noise, volatile=True)
	gen_image = generator(noise).view(-1, nc, im_size, im_size)
	#gen_image = denorm(gen_image)
	gen_image = gen_image

	#n_rows = np.sqrt(noise.size()[0]).astype(np.int32)
	#n_cols = np.sqrt(noise.size()[0]).astype(np.int32)
	n_cols, n_rows = fig_size
	fig, axes = plt.subplots(n_cols, n_rows, figsize=(n_rows, n_cols))
	for ax, img in zip(axes.flatten(), gen_image):
		ax.axis('off')
		ax.set_adjustable('box-forced')

		img = img.cpu().data

		if enhance:
			img_E = ImageEnhance.Sharpness( to_pil(img) ).enhance(1.)
			img = to_tensor(img_E)

		# Scale to 0-255
		img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8).squeeze()
		# ax.imshow(img.cpu().data.view(image_size, image_size, 3).numpy(), cmap=None, aspect='equal')

		if nc == 1:
			ax.imshow(img, cmap="gray", aspect='equal')
		else:
			ax.imshow(img, cmap=None, aspect='equal')	

	plt.subplots_adjust(wspace=0, hspace=0)
	#title = 'Samples'
	#fig.text(0.5, 0.04, title, ha='center')

	# save figure

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	save_fn = save_dir + prefix + '_' + cp_name + '.pdf'
	plt.savefig(save_fn)

	plt.close()

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--n-tests', type=int, default=4, metavar='N', help='number of samples to generate (default: 4)')
	parser.add_argument('--no-plots', action='store_true', default=False, help='Disables plot of train/test losses')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	model = Generator()

	ckpt = torch.load(args.cp_path, map_location=lambda storage, loc: storage)
	model.load_state_dict(ckpt['model_state'])

	if args.cuda:
		model = model.cuda()

	print('Cuda Mode is: {}'.format(args.cuda))

	history = ckpt['history']

	if not args.no_plots:
		plot_learningcurves(history, list(history.keys()))

	test_model(model=model, n_tests=args.n_tests, cuda_mode=args.cuda)
	save_samples(generator=model, cp_name=args.cp_path.split('/')[-1].split('.')[0], prefix='mnist', fig_size=(30, 30), nc=1, im_size=30, cuda_mode=args.cuda, enhance=False)
