from __future__ import print_function
import argparse
import torch
import models_zoo
from cup_generator.model import Generator

# Training settings
parser = argparse.ArgumentParser(description='test models')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--n-frames', type=int, default=25, metavar='N', help='Number of frames per sample (default: 128)')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

model = models_zoo.model_gen(n_frames=args.n_frames, cuda_mode=args.cuda)
generator = Generator().eval()
frames_list=[]

if args.cuda:
	model = model.cuda()
	generator = generator.cuda()

dummy_input = torch.rand(10, 1, 256, 355)

out_seq = model(dummy_input)

for i in range(out_seq.size(1)):
	gen_frame = generator(out_seq[:,i,:].squeeze().contiguous())

print(out_seq.size(), gen_frame.size())
