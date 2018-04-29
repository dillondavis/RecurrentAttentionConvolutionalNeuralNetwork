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



class RankLoss(nn.Module):
    def __init__(self, margin):
        super(RankLoss, self).__init__()
        self.margin = margin

    def forward(self, scores1, scores2, target):
        ps1 = F.softmax(scores1)[:, target.long().data]
        ps2 = F.softmax(scores2)[:, target.long().data]
        return torch.clamp(ps1 - ps2 + self.margin, min=0)


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
        self.train_data_loader = train_loader(
            args.train_path, args.batch_size, pin_memory=args.cuda)
        self.test_data_loader = test_loader(
            args.test_path, args.batch_size, pin_memory=args.cuda)
        self.criterion_class = nn.CrossEntropyLoss()
        self.criterion_rank = RankLoss(0.05)

    def eval(self):
        """Performs evaluation."""
        self.model.eval()
        error_meters = None

        print('Performing eval...')
        for batch, label in tqdm(self.test_data_loader, desc='Eval'):
            if self.cuda:
                batch = batch.cuda()
            batch = Variable(batch, volatile=True)
            scores = self.model(batch)
            # Init error meter.
            outputs = [score.data.view(-1, score.size(1)) for score in scores]
            label = label.view(-1)
            if error_meters is None:
                topk = [1]
                if outputs[0].size(1) > 5:
                    topk.append(5)
                error_meters = [tnt.meter.ClassErrorMeter(topk=topk) for _ in range(self.args.scale)]
            for error_meter, output in zip(error_meters, outputs):
                error_meter.add(output, label)

        errors = [error_meter.value() for error_meter in error_meters]
        for i, error in enumerate(errors):
            print('Scale {} Error: '.format(i+1) + ', '.join('@%s=%.2f' % t for t in zip(topk, error)))
        self.model.train()

        return errors

    def do_batch(self, optimizer, batch, label, optimize_class=True):
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
        label = Variable(label)

        # Set grads to 0.
        self.model.zero_grad()

        # Do forward-backward.
        scores = self.model(batch)
        if optimize_class:
            for i in range(len(scores)-1, -1, -1): 
                if optimize_class:
                    retain_graph = i > 0
                    self.criterion_class(scores[i], label).backward(retain_graph=retain_graph)
       	else: 
            for i in range(len(scores)-1, 0, -1): 
                retain_graph = (i-1) > 0
                self.criterion_rank(scores[i-1], scores[i], label).backward(retain_graph=retain_graph)
                
        # Update params.
        optimizer.step()

    def do_epoch(self, epoch_idx, optimizer, optimize_class=True):
        """
        Trains model for one epoch
        :param epoch_idx: int epoch number
        :param optimizer: Optimizer for training
        """
        for batch, label in tqdm(self.train_data_loader, desc='Epoch: %d ' % (epoch_idx)):
            self.do_batch(optimizer, batch, label, optimize_class=optimize_class)

    def save_model(self, epoch, best_accuracy, errors, savename):
        """Saves model to file."""
        # Prepare the ckpt.
        self.model.cpu()
        ckpt = {
            'args': self.args,
            'epoch': epoch,
            'accuracy': best_accuracy,
            'errors': errors,
            'state_dict': self.model.state_dict(),
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
        self.model.load_state_dict(ckpt['state_dict'])
        self.args = ckpt['args']

    def train(self, epochs, cnn_optimizer, apn_optimizer, savename='', best_accuracy=0):
        """Performs training."""
        best_accuracy = best_accuracy
        error_history = []
        optimize_class = True
        class_epoch = 0
        rank_epoch = 0

        if self.args.cuda:
            self.model = self.model.cuda()

        for i in range(epochs):
            print('Epoch : {}'.format(i+1))
            epoch_idx = (class_epoch if optimize_class else rank_epoch) + 1
            epoch_type = 'Class' if optimize_class else 'Rank'
            print('Optimize {} Epoch: {}'.format(epoch_type, epoch_idx))

            optimizer = cnn_optimizer if optimize_class else apn_optimizer
            optimizer.update_lr(epoch_idx)
            self.model.train()
            self.do_epoch(epoch_idx, optimizer, optimize_class=optimize_class)
            errors = self.eval()
            accuracy = 100 - errors[-1][0]  # Top-1 accuracy.
            error_history.append(errors)

            # Save performance history and stats.
            with open(savename + '.json', 'w') as fout:
                json.dump({
                    'error_history': error_history,
                    'args': vars(self.args),
                }, fout)

            if optimize_class:
                class_epoch += 1
            else:
                rank_epoch += 1

            if (accuracy - best_accuracy) < self.args.converge_acc_diff:
                optimize_class = not optimize_class
            # Save best model, if required.
            if accuracy > best_accuracy:
                print('Best model so far, Accuracy: %0.2f%% -> %0.2f%%' %
                      (best_accuracy, accuracy))
                best_accuracy = accuracy
                self.save_model(epoch_idx, best_accuracy, errors, savename)


        print('Finished finetuning...')
        print('Best error/accuracy: %0.2f%%, %0.2f%%' %
              (100 - best_accuracy, best_accuracy))
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
