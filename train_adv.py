from __future__ import print_function
import argparse
import torch
import models_zoo
from data_load import Loader
from train_loop_adv import TrainLoop
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

# Training settings
parser = argparse.ArgumentParser(description='Online transfer learning for emotion recognition tasks')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--patience', type=int, default=30, metavar='N', help='number of epochs without improvement to wait before stopping training (default: 30)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--l2', type=float, default=0.000001, metavar='lambda', help='L2 wheight decay coefficient (default: 0.00001)')
parser.add_argument('--ngpus', type=int, default=0, help='Number of GPUs to use. Default=0 (no GPU)')
parser.add_argument('--input-data-path', type=str, default='./data/input/', metavar='Path', help='Path to data input data')
parser.add_argument('--targets-data-path', type=str, default='./data/targets/', metavar='Path', help='Path to output data')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=None, metavar='N', help='how many batches to wait before logging training status. If None, cp is done every epoch')
parser.add_argument('--n_workers', type=int, default=1)
args = parser.parse_args()
args.cuda = True if args.ngpus>0 and torch.cuda.is_available() else False

train_data_set = Loader(input_file=args.input_data_path+'input_train.hdf', output_file=args.targets_data_path+'output_train.hdf')
valid_data_set = Loader(input_file=args.input_data_path+'input_valid.hdf', output_file=args.targets_data_path+'output_valid.hdf')

train_loader = DataLoader(train_data_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
valid_loader = DataLoader(valid_data_set, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.n_workers)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#model = models_zoo.model(args.cuda)
model = models_zoo.small_model(args.cuda)
discriminator = models_zoo.discriminator()

if args.ngpus > 1:
	model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpus)))
	discriminator = torch.nn.DataParallel(discriminator, device_ids=list(range(args.ngpus)))

if args.cuda:
	model = model.cuda()
	discriminator = discriminator.cuda()

optimizer_g = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, weight_decay=args.l2)

trainer = TrainLoop(model, discriminator, optimizer_g, optimizer_d, train_loader, valid_loader, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda)

print('Cuda Mode is: {}'.format(args.cuda))

trainer.train(n_epochs=args.epochs, patience = args.patience)
