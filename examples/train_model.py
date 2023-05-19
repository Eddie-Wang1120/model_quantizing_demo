import traceback
import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
import argparse
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from quantizing.models import create_models
from quantizing.data_loggers import *
import quantizing.apputils as apputils
import torchnet.meter as tnt
import math
from collections import OrderedDict
import operator
from quantizing.utils import *
msglogger = logging.getLogger()

def float_range(min_val=0., max_val=1., exc_min=False, exc_max=False):
    def checker(val_str):
        val = float(val_str)
        min_op, min_op_str = (operator.gt, '>') if exc_min else (operator.ge, '>=')
        max_op, max_op_str = (operator.lt, '<') if exc_max else (operator.le, '<=')
        if min_op(val, min_val) and max_op(val, max_val):
            return val
        raise argparse.ArgumentTypeError(
            'Value must be {} {} and {} {} (received {})'.format(min_op_str, min_val, max_op_str, max_val, val))
    if min_val >= max_val:
        raise ValueError('min_val must be less than max_val')
    return checker

def init_dataset(args):
    if not hasattr(args, 'dataset'):
        if 'cifar' in args.arch:
            dataset = 'cifar10' 
        elif 'mnist' in args.arch:
            dataset = 'mnist' 
        else:
            dataset = 'imagenet'
    args.dataset = dataset
    return args

def init_logger(args):
    global msglogger
    timestr = time.strftime("%Y.%m.%d-%H%M%S")
    exp_full_name = timestr
    logdir = os.path.join(args.output_dir, exp_full_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log_filename = os.path.join(logdir, exp_full_name + '.log')
    apply_default_logger_cfg(log_filename)
    msglogger.logdir = logdir
    msglogger.log_filename = log_filename
    if args.verbose:
        msglogger.setLevel(logging.DEBUG)
    msglogger.info('Log file for this run: ' + os.path.realpath(log_filename))
    return msglogger.logdir

def apply_default_logger_cfg(log_filename):
    d = {
        'version': 1,
        'formatters': {
            'simple': {
                'class': 'logging.Formatter',
                'format': '%(asctime)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': log_filename,
                'mode': 'w',
                'formatter': 'simple',
            },
        },
        'loggers': {
            '': {  # root logger
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'app_cfg': {
                'level': 'DEBUG',
                'handlers': ['file'],
                'propagate': False
            },
        }
    }

    logging.config.dictConfig(d)


def init_learner(args):
    model = create_models(args.dataset, args.arch)
    optimizer = None
    start_epoch = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    msglogger.debug('Optimizer Type: %s', type(optimizer))
    msglogger.debug('Optimizer Args: %s', optimizer.defaults)
    return model, optimizer, start_epoch, args.epochs

def load_data(args, fixed_subset=False, sequential=False, load_train=True, load_val=True, load_test=True):
    test_only = not load_train and not load_val

    train_loader, val_loader, test_loader, _ = apputils.load_data(args.dataset, args.arch,
                              os.path.expanduser(args.data), args.batch_size,
                              args.workers, args.validation_split, args.deterministic,
                              args.effective_train_size, args.effective_valid_size, args.effective_test_size,
                              fixed_subset, sequential, test_only)
    if test_only:
        msglogger.info('Dataset sizes:\n\ttest=%d', len(test_loader.sampler))
    else:
        msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                       len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

    loaders = (train_loader, val_loader, test_loader)
    flags = (load_train, load_val, load_test)
    loaders = [loaders[i] for i, flag in enumerate(flags) if flag]
    
    if len(loaders) == 1:
        # Unpack the list for convenience
        loaders = loaders[0]
    return loaders

def train(train_loader, model, criterion, optimizer, epoch, loggers, args):
    """Training-with-compression loop for one epoch.
    
    For each training step in epoch:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input)
        loss = criterion(output, target)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        compression_scheduler.before_parameter_optimization(epoch)
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)
    """
    def _log_training_progress():
        # Log some statistics
        errs = OrderedDict()
        errs['Top1'] = classerr.value(1)
        errs['Top5'] = classerr.value(5)
        stats_dict = OrderedDict()
        for loss_name, meter in losses.items():
            stats_dict[loss_name] = meter.mean
        stats_dict.update(errs)
        stats_dict['LR'] = optimizer.param_groups[0]['lr']
        stats_dict['Time'] = batch_time.mean
        stats = ('Performance/Training/', stats_dict)

        params = model.named_parameters() if args.log_params_histograms else None
        log_training_progress(stats,
                                        params,
                                        epoch, steps_completed,
                                        steps_per_epoch, args.print_freq,
                                        loggers)
    OVERALL_LOSS_KEY = 'Overall Loss'
    OBJECTIVE_LOSS_KEY = 'Objective Loss'

    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    msglogger.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)
    
    # Switch to train mode
    model.train()
    acc_stats = []
    end = time.time()
    for train_step, (inputs, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.add(time.time() - end)
        inputs, target = inputs.to(args.device), target.to(args.device)

        output = model(inputs)

        loss = criterion(output, target)
            # Measure accuracy
            # For inception models, we only consider accuracy of main classifier
        if isinstance(output, tuple):
            classerr.add(output[0].detach(), target)
        else:
            classerr.add(output.detach(), target)
        acc_stats.append([classerr.value(1), classerr.value(5)])
        # Record loss
        losses[OBJECTIVE_LOSS_KEY].add(loss.item())

        losses[OVERALL_LOSS_KEY].add(loss.item())

        # Compute the gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        batch_time.add(time.time() - end)
        steps_completed = (train_step+1)

        if steps_completed % args.print_freq == 0:
            _log_training_progress()
        end = time.time()
    #return acc_stats
    # NOTE: this breaks previous behavior, which returned a history of (top1, top5) values
    return classerr.value(1), classerr.value(5), losses[OVERALL_LOSS_KEY]

def _validate(data_loader, model, criterion, loggers, args, epoch=-1):
    """Execute the validation/test loop."""
    losses = {'objective_loss': tnt.AverageValueMeter()}
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))

    batch_time = tnt.AverageValueMeter()
    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    total_steps = total_samples / batch_size
    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

    # Switch to evaluation mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for validation_step, (inputs, target) in enumerate(data_loader):
            inputs, target = inputs.to(args.device), target.to(args.device)
            # compute output from model
            output = model(inputs)

            # compute loss
            loss = criterion(output, target)
            # measure accuracy and record loss
            losses['objective_loss'].add(loss.item())
            classerr.add(output.detach(), target)
            # measure elapsed time
            batch_time.add(time.time() - end)
            end = time.time()

            steps_completed = (validation_step+1)

    msglogger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                    classerr.value()[0], classerr.value()[1], losses['objective_loss'].mean)

    return classerr.value(1), classerr.value(5), losses['objective_loss'].mean


def validate(val_loader, model, criterion, loggers, args, epoch=-1):
    """Model validation"""
    if epoch > -1:
        msglogger.info('--- validate (epoch=%d)-----------', epoch)
    else:
        msglogger.info('--- validate ---------------------')
    return _validate(val_loader, model, criterion, loggers, args, epoch)


def test(test_loader, model, criterion, loggers=None, activations_collectors=None, args=None):
    """Model Test"""
    msglogger.info('--- test ---------------------')

    top1, top5, lossses = _validate(test_loader, model, criterion, loggers, args)

    return top1, top5, lossses


class NaiveModel(object):
    def __init__(self, args):
        self.args = args
        args = init_dataset(self.args)
        self.logdir = init_logger(args)
        self.tflogger = TensorBoardLogger(msglogger.logdir)
        self.pylogger = PythonLogger(msglogger)
        self.model, self.optimizer, self.start_epoch, self.ending_epoch = init_learner(args)
        self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        self.train_loader, self.val_loader, self.test_loader = (None, None, None)
        self.performance_tracker = apputils.SparsityAccuracyTracker(self.args.num_best_scores)

    def load_datasets(self):
        """Load the datasets"""
        if not all((self.train_loader, self.val_loader, self.test_loader)):
            self.train_loader, self.val_loader, self.test_loader = load_data(self.args)
        return self.data_loaders

    @property
    def data_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def train_one_epoch(self, epoch, verbose=True):
        """Train for one epoch"""
        self.load_datasets()
        top1, top5, loss = train(self.train_loader, self.model, self.criterion, self.optimizer, 
                                     epoch, loggers=[self.tflogger, self.pylogger], args=self.args)
        return top1, top5, loss

    def validate_one_epoch(self, epoch, verbose=True):
        """Evaluate on validation set"""
        self.load_datasets()
        
        top1, top5, vloss = validate(self.val_loader, self.model, self.criterion, 
                                         [self.pylogger], self.args, epoch)
        return top1, top5, vloss

    def train_validate_with_scheduling(self, epoch, validate=True, verbose=True):
        top1, top5, loss = self.train_one_epoch(epoch, verbose)
        if validate:
            top1, top5, loss = self.validate_one_epoch(epoch, verbose)
            
        return top1, top5, loss

    def _finalize_epoch(self, epoch, top1, top5):
        # Update the list of top scores achieved so far, and save the checkpoint
        self.performance_tracker.step(self.model, epoch, top1=top1, top5=top5)
        best_score = self.performance_tracker.best_scores()[0]
        is_best = epoch == best_score.epoch
        checkpoint_extras = {'current_top1': top1,
                             'best_top1': best_score.top1,
                             'best_epoch': best_score.epoch}
        if msglogger.logdir:
            apputils.save_checkpoint(epoch, self.args.arch, self.model, optimizer=self.optimizer,
                                     scheduler=None, extras=checkpoint_extras,
                                     is_best=is_best, name=self.args.name, dir=msglogger.logdir)


    def begin_train_loop(self):
        self.load_datasets()
        self.performance_tracker.reset()
        for epoch in range(self.start_epoch, self.ending_epoch):
            top1, top5, loss = self.train_validate_with_scheduling(epoch)

            self._finalize_epoch(epoch, top1, top5)
        return self.performance_tracker.perf_scores_history
    
    def test(self):
        self.load_datasets()
        return test(self.test_loader, self.model, self.criterion,
                    self.pylogger, args=self.args)

        
        
def train_parse():
    parser = argparse.ArgumentParser(description="train a model before quant")
    parser.add_argument('data', help='path to dataset')
    parser.add_argument('--arch', help='model architectures')
    parser.add_argument('--epochs', type=int, metavar='N', default=90,
                        help='number of total epochs to run (default: 90')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--out-dir', '-o', dest='output_dir', default='logs', help='Path to dump logs and checkpoints')
    parser.add_argument('--verbose', '-v', action='store_true', help='Emit debug log messages')
    parser.add_argument('--device', default='cuda', help='cpu or cuda')
    parser.add_argument('--num-best-scores', dest='num_best_scores', default=1, type=int,
                        help='number of best scores to track and report (default: 1)')
    parser.add_argument('--validation-split', '--valid-size', '--vs', dest='validation_split',
                        type=float_range(exc_max=True), default=0.1,
                        help='Portion of training dataset to set aside for validation')
    parser.add_argument('--deterministic', '--det', action='store_true',
                        help='Ensure deterministic execution for re-producible results.')
    parser.add_argument('--effective-train-size', '--etrs', type=float_range(exc_min=True), default=1.,
                        help='Portion of training dataset to be used in each epoch. '
                             'NOTE: If --validation-split is set, then the value of this argument is applied '
                             'AFTER the train-validation split according to that argument')
    parser.add_argument('--effective-valid-size', '--evs', type=float_range(exc_min=True), default=1.,
                        help='Portion of validation dataset to be used in each epoch. '
                             'NOTE: If --validation-split is set, then the value of this argument is applied '
                             'AFTER the train-validation split according to that argument')
    parser.add_argument('--effective-test-size', '--etes', type=float_range(exc_min=True), default=1.,
                        help='Portion of test dataset to be used in each epoch')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--param-hist', dest='log_params_histograms', action='store_true', default=False,
                        help='log the parameter tensors histograms to file '
                             '(WARNING: this can use significant disk space)')
    parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')

    optimizer_args = parser.add_argument_group('Optimizer arguments')
    optimizer_args.add_argument('--lr', '--learning-rate', default=0.1,
                    type=float, metavar='LR', help='initial learning rate')
    optimizer_args.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum')
    optimizer_args.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

    args = parser.parse_args()
    return args

def main():
    args = train_parse()
    model = NaiveModel(args)
    model.begin_train_loop()
    return model.test()
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None and hasattr(msglogger, 'log_filename'):
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))