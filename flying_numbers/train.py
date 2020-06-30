from __future__ import print_function
import argparse
import torch
import models_zoo
from cup_generator.model import Generator
from data_load import Loader
from train_loop import TrainLoop
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from optimizer import TransformerOptimizer
from torch.utils.tensorboard import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='CUP reconstruction')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=128, metavar='N', help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='LR', help='learning rate (default: 0.0002)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='m', help='Momentum paprameter (default: 0.9)')
parser.add_argument('--max-gnorm', type=float, default=10., metavar='clip', help='Max gradient norm (default: 10.0)')
parser.add_argument('--warmup', type=int, default=4000, metavar='N', help='Iterations until reach lr (default: 4000)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--generator-path', type=str, default=None, metavar='Path', help='Path for generator params')
parser.add_argument('--pretrained-path', type=str, default=None, metavar='Path', help='Path to trained model. Discards output layer')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=5, metavar='N', help='how many batches to wait before logging training status. (default: 5)')
parser.add_argument('--n-workers', type=int, default=2)
parser.add_argument('--logdir', type=str, default=None, metavar='Path', help='Path for checkpointing')
### Data options
parser.add_argument('--im-size', type=int, default=64, metavar='N', help='H and W of frames (default: 64)')
parser.add_argument('--n-digits', type=int, default=2, metavar='N', help='Number of bouncing digits (default: 2)')
parser.add_argument('--n-frames', type=int, default=40, metavar='N', help='Number of frames per sample (default: 40)')
parser.add_argument('--rep-times', type=int, default=1, metavar='N', help='Number of times consecutive frames are repeated. No rep is equal to 1 (default: 1)')
parser.add_argument('--train-examples', type=int, default=50000, metavar='N', help='Number of training examples (default: 50000)')
parser.add_argument('--val-examples', type=int, default=5000, metavar='N', help='Number of validation examples (default: 500)')
parser.add_argument('--mask-path', type=str, default=None, metavar='Path', help='path to encoding mask')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

if args.logdir:
	writer = SummaryWriter(log_dir=args.logdir, comment='reconstruction', purge_step=True if args.checkpoint_epoch is None else False)
else:
	writer = None

train_data_set = Loader(im_size=args.im_size, n_objects=args.n_digits, n_frames=args.n_frames, rep_times=args.rep_times, sample_size=args.train_examples, mask_path=args.mask_path)
valid_data_set = Loader(im_size=args.im_size, n_objects=args.n_digits, n_frames=args.n_frames, rep_times=args.rep_times, sample_size=args.val_examples, mask_path=args.mask_path)

train_loader = DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
valid_loader = DataLoader(valid_data_set, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.n_workers)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = models_zoo.model_gen(n_frames=train_data_set.n_frames, cuda_mode=args.cuda)
generator = Generator().eval()

gen_state = torch.load(args.generator_path, map_location=lambda storage, loc: storage)
print(generator.load_state_dict(gen_state['model_state'], strict=True))

if args.pretrained_path:
	print('\nLoading pretrained model from: {}\n'.format(args.pretrained_path))
	ckpt=torch.load(args.pretrained_path, map_location = lambda storage, loc: storage)
	print(model.load_state_dict(ckpt['model_state'], strict=False))
	print('\n')

if args.cuda:
	model = model.cuda()
	generator = generator.cuda()
	torch.backends.cudnn.benchmark=True

print(model)
print('\n')
print(generator)
print('\n')

optimizer = TransformerOptimizer(optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2, nesterov=True), 
			lr=args.lr, 
			warmup_steps=args.warmup)

trainer = TrainLoop(model, generator, optimizer, train_loader, valid_loader, max_gnorm=args.max_gnorm, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda, logger=writer)

args_dict = dict(vars(args))
for key in args_dict:
	print('{}: {}'.format(key, args_dict[key]))
print('\n')

trainer.train(n_epochs=args.epochs, save_every = args.save_every)
