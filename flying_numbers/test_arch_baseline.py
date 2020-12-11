from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import models_zoo
from pytorch_msssim import ms_ssim

# Training settings
parser = argparse.ArgumentParser(description='test models')
parser.add_argument('--n-frames', type=int, default=40, metavar='N', help='Number of frames per sample (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

print(args)

model = models_zoo.model_baseline(n_frames=args.n_frames)
dummy_input = torch.rand(10, 1, 64, 64, args.n_frames)

if args.cuda:
	model = model.cuda()
	dummy_input = dummy_input.cuda()

out_rec = model(dummy_input)

print(out_rec.size())

for i in range(out_rec.size(-1)):
	gen_frame = out_rec[...,i]
	print(i, gen_frame.unsqueeze(1).size())
	print(ms_ssim(F.upsample(gen_frame, scale_factor=3), F.upsample(out_rec[...,i], scale_factor=3)))