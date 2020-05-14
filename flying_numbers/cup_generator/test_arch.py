from __future__ import print_function

import argparse
import os
import sys
import torch
import torch.optim as optim

from model import *

parser = argparse.ArgumentParser(description='Hyper volume training of GANs')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False


generator = Generator()
disc = Discriminator(optim.Adam, 0.1, (0.1, 0.1))

dummy_input = torch.rand(3,128,1,1)

gen_sample = generator(dummy_input)

disc_pred = disc(gen_sample)

print(dummy_input.size(), gen_sample.size(), disc_pred.size())