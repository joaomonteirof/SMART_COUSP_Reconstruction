from __future__ import print_function
import argparse
import torch
import models_zoo
from data_load import Loader, Loader_manyfiles
from train_loop_adv import TrainLoop
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

# Training settings
parser = argparse.ArgumentParser(description='Online transfer learning for emotion recognition tasks')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--valid-batch-size', type=int, default=256, metavar='N', help='input batch size for testing (default: 512)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--patience', type=int, default=10, metavar='N', help='How many epochs without improvement to wait before reducing the LR (default: 10)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='LR', help='learning rate (default: 0.0002)')
parser.add_argument('--beta1', type=float, default=0.5, metavar='beta1', help='Adam beta 1 (default: 0.5)')
parser.add_argument('--beta2', type=float, default=0.999, metavar='beta2', help='Adam beta 2 (default: 0.99)')
parser.add_argument('--ndiscriminators', type=int, default=8, help='Number of discriminators. Default=8')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--input-data-path', type=str, default='./data/input/', metavar='Path', help='Path to data input data')
parser.add_argument('--targets-data-path', type=str, default='./data/targets/', metavar='Path', help='Path to output data')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=5, metavar='N', help='how many batches to wait before logging training status. (default: 5)')
parser.add_argument('--n-workers', type=int, default=2)
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

#train_data_set = Loader(input_file=args.input_data_path+'input_train.hdf', output_file=args.targets_data_path+'output_train.hdf')
train_data_set = Loader_manyfiles(input_file_base_name=args.input_data_path+'input_train', output_file_base_name=args.targets_data_path+'output_train', n_files=4)
valid_data_set = Loader(input_file=args.input_data_path+'input_valid.hdf', output_file=args.targets_data_path+'output_valid.hdf')

train_loader = DataLoader(train_data_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
valid_loader = DataLoader(valid_data_set, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.n_workers)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = models_zoo.model_3d_lstm(args.cuda)

disc_list = []
for i in range(args.ndiscriminators):
	disc = models_zoo.discriminator_proj(optim.Adam, args.lr, (args.beta1, args.beta2)).train()
	disc_list.append(disc)

if args.cuda:
	model = model.cuda()
	for disc in disc_list:
		disc = disc.cuda()
	torch.backends.cudnn.benchmark=True

optimizer_g = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

trainer = TrainLoop(model, disc_list, optimizer_g, train_loader, valid_loader, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda)

print('Cuda Mode is: {}'.format(args.cuda))
print('Number of discriminators is: {}'.format(len(disc_list)))

trainer.train(n_epochs=args.epochs, patience = args.patience, save_every = args.save_every)
