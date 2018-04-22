"""Training manager and multi optimizer for training PyTorch models"""
import json
import torch
import dataset
import utils
import torch.nn as nn
import torch.nn.functional as F
import torchnet as tnt
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable


class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, model):
        """
        Initializes training manager with training args and model
        :param args: argparse training args from command line
        :param model: PyTorch model to train
        """
        self.args = args

        self.cuda = args.cuda
        self.model = model

        # Set up data loader, criterion, and pruner.
        train_loader = dataset.train_loader_cubs
        test_loader = dataset.test_loader_cubs
        self.train_data_loader = train_loader(args.train_path, 
	    args.batch_size, pin_memory=args.cuda, coords=True)
        self.test_data_loader = test_loader(args.test_path, 
            args.batch_size, pin_memory=args.cuda, coords=True)
        self.criterion = nn.MSELoss()

    def eval(self):
        """Performs evaluation."""
        self.model.eval()
        error_meter = None

        print('Performing eval...')
        for batch, label in tqdm(self.test_data_loader, desc='Eval'):
            if self.cuda:
                batch = batch.cuda()
                label = label.cuda()
            batch = Variable(batch, volatile=True)
            label = Variable(label, volatile=True).float()
            output = self.model(batch, label)

            # Init error meter.
            if error_meter is None:
                error_meter = tnt.meter.MSEMeter()
            label = label.data.cpu().numpy()
            output = output.data.cpu().numpy()
            error_meter.add(output, label)

        error = error_meter.value()
        print('MSE: {}'.format(error))
        self.model.train()

        return error

    def do_batch(self, optimizer, batch, label):
        """
        Runs model for one batch
        :param optimizer: Optimizer for training
        :param batch: (num_batch, 3, h, w) Torch tensor of data
        :param label: (num_batch) Torch tensor of classes
        """
        if self.cuda:
            batch = batch.cuda()
            label = label.cuda()
        batch = Variable(batch)
        label = Variable(label).float()

        # Set grads to 0.
        self.model.zero_grad()

        # Do forward-backward.
        output = self.model(batch, label)
        self.criterion(output, label).backward()

        # Update params.
        # nn.utils.clip_grad.clip_grad_norm(self.model.parameters(), 1)
        optimizer.step()

    def do_epoch(self, epoch_idx, optimizer):
        """
        Trains model for one epoch
        :param epoch_idx: int epoch number
        :param optimizer: Optimizer for training
        """
        for batch, label in tqdm(self.train_data_loader, desc='Epoch: %d ' % (epoch_idx)):
            self.do_batch(optimizer, batch, label)

    def save_model(self, epoch, best_accuracy, errors, savename):
        """Saves model to file."""
        # Prepare the ckpt.
        self.model.cpu()
        ckpt = {
            'args': self.args,
            'epoch': epoch,
            'accuracy': best_accuracy,
            'errors': errors,
            'apn1_state_dict': self.model.apn1.state_dict(),
            'apn2_state_dict': self.model.apn2.state_dict(),
        }
        if self.cuda:
            self.model.cuda()

        # Save to file.
        torch.save(ckpt, savename + '.pt')

    def load_model(self, savename):
        """
        Loads model from a saved model pt file
        :param savename: string file prefix
        """
        ckpt = torch.load(savename +'.pt')
        self.model.apn1.load_state_dict(ckpt['apn1_state_dict'])
        self.model.apn2.load_state_dict(ckpt['apn2_state_dict'])
        self.args = ckpt['args']

    def train(self, epochs, optimizer, savename=''):
        """Performs training."""
        best_error = np.inf
        error_history = []

        if self.args.cuda:
            self.model = self.model.cuda()

        for i in range(epochs):
            epoch_idx = i + 1
            optimizer.update_lr(epoch_idx)
            self.model.train()
            self.do_epoch(epoch_idx, optimizer)
            error = self.eval()
            error_history.append(error)

            # Save performance history and stats.
            with open(savename + '.json', 'w') as fout:
                json.dump({
                    'error_history': error_history,
                    'args': vars(self.args),
                }, fout)

            # Save best model, if required.
            if error < best_error:
                print('Best model so far, Error: %0.2f%% -> %0.2f%%' %
                      (best_error, error))
                best_error = error
                self.save_model(epoch_idx, best_error, error, savename)


        print('Finished finetuning...')
        print('Best error: %0.2f%%' %
              (best_error))
        print('-' * 16)


class Optimizers(object):
    """Handles a list of optimizers."""

    def __init__(self, args):
        self.optimizers = []
        self.lrs = []
        self.decay_every = []
        self.args = args

    def add(self, optimizer, learning_rate, decay_every):
        """Adds optimizer to list."""
        self.optimizers.append(optimizer)
        self.lrs.append(learning_rate)
        self.decay_every.append(decay_every)

    def step(self):
        """Makes all optimizers update their params."""
        for optimizer in self.optimizers:
            optimizer.step()

    def update_lr(self, epoch_idx):
        """Update learning rate of every optimizer."""
        for optimizer, init_lr, decay_every in zip(self.optimizers, self.lrs, self.decay_every):
            optimizer = utils.step_lr(
                epoch_idx, init_lr, decay_every,
                self.args.lr_decay_factor, optimizer
            )
