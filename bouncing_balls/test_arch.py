from __future__ import print_function
import argparse
import torch
import models_zoo
from cup_generator.model import Generator
import time

# Training settings
parser = argparse.ArgumentParser(description='test models')
parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='Batch size (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--n-frames', type=int, default=100, metavar='N', help='Number of frames per sample (default: 100)')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

model = models_zoo.model_gen(n_frames=args.n_frames, cuda_mode=args.cuda)
generator = Generator().eval()
frames_list=[]

if args.cuda:
	model = model.cuda().eval()
	generator = generator.cuda().eval()

dummy_input = torch.rand(args.batch_size, 1, 256, 355).to(next(model.parameters()).device)

start = time.time()

with torch.no_grad():

	out_seq = model(dummy_input)

	end_single_frame = time.time()

	for i in range(out_seq.size(1)):
		gen_frame = generator(out_seq[:,i,:].squeeze(1).contiguous())

end_full_reconstruction = time.time()

print(out_seq.size(), gen_frame.size())

print('Elapsed time - Fisrt frame: {:0.4f}ms'.format((end_single_frame-start)*1000.0))
print('Elapsed time - Full reconstruction: {:0.4f}ms'.format((end_full_reconstruction-start)*1000.0))