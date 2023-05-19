# python3 quantize_model.py --arch simplenet_cifar ../data.cifar10 --resume trained_models/checkpoint.pth.tar --quantize-eval --evaluate

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
import copy
from quantizing.quantization import *
from quantizing.scheduler import *

from functools import reduce, partial, update_wrapper

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

def init_learner(args):
    model = create_models(args.dataset, args.arch)
    compression_scheduler = None
    start_epoch = 0
    if args.resumed_checkpoint_path:
        model, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(
            model, args.resumed_checkpoint_path, model_device=args.device)    
    elif args.load_model_path:
        model = apputils.load_lean_checkpoint(model, args.load_model_path, model_device=args.device)
    if compression_scheduler is None:
        compression_scheduler = CompressionScheduler(model)

    return model, compression_scheduler, start_epoch, args.epochs

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

def test(test_loader, model, criterion, loggers=None, activations_collectors=None, args=None):
    """Model Test"""
    msglogger.info('--- test ---------------------')

    top1, top5, lossses = _validate(test_loader, model, criterion, loggers, args)

    return top1, top5, lossses

def evaluate_model(test_loader, model, criterion, loggers, activations_collectors=None, args=None, scheduler=None):
    # This sample application can be invoked to evaluate the accuracy of your model on
    # the test dataset.
    # You can optionally quantize the model to 8-bit integer before evaluation.
    # For example:
    # python3 compress_classifier.py --arch resnet20_cifar  ../data.cifar10 -p=50 --resume-from=checkpoint.pth.tar --evaluate

    if not isinstance(loggers, list):
        loggers = [loggers]

    if not args.quantize_eval:
        # Handle case where a post-train quantized model was loaded, and user wants to convert it to PyTorch
        return test(test_loader, model, criterion, loggers, activations_collectors, args=args)
    else:
        return quantize_and_test_model(test_loader, model, criterion, args, loggers,
                                       scheduler=scheduler, save_flag=True)


def quantize_and_test_model(test_loader, model, criterion, args, loggers=None, scheduler=None, save_flag=True):
    """Collect stats using test_loader (when stats file is absent),

    clone the model and quantize the clone, and finally, test it.
    args.device is allowed to differ from the model's device.
    When args.qe_calibration is set to None, uses 0.05 instead.

    scheduler - pass scheduler to store it in checkpoint
    save_flag - defaults to save both quantization statistics and checkpoint.
    """
    if hasattr(model, 'quantizer_metadata') and \
            model.quantizer_metadata['type'] == quantizing.quantization.PostTrainLinearQuantizer:
        raise RuntimeError('Trying to invoke post-training quantization on a model that has already been post-'
                           'train quantized. Model was likely loaded from a checkpoint. Please run again without '
                           'passing the --quantize-eval flag')
    if not (args.qe_dynamic or args.qe_stats_file or args.qe_config_file):
        args_copy = copy.deepcopy(args)
        args_copy.qe_calibration = args.qe_calibration if args.qe_calibration is not None else 0.05

        # set stats into args stats field
        args.qe_stats_file = acts_quant_stats_collection(
            model, criterion, loggers, args_copy, save_to_file=save_flag)

    args_qe = copy.deepcopy(args)
    qe_model = copy.deepcopy(model).to(args.device)

    quantizer = quantizing.quantization.PostTrainLinearQuantizer.from_args(qe_model, args_qe)
    dummy_input = quantizing.get_dummy_input(input_shape=model.input_shape)
    quantizer.prepare_model(dummy_input)

    if args.qe_convert_pytorch:
        qe_model = _convert_ptq_to_pytorch(qe_model, args_qe)

    test_res = test(test_loader, qe_model, criterion, loggers, args=args_qe)

    if save_flag:
        checkpoint_name = 'quantized'
        apputils.save_checkpoint(0, args_qe.arch, qe_model, scheduler=scheduler,
            name='_'.join([args_qe.name, checkpoint_name]) if args_qe.name else checkpoint_name,
            dir=msglogger.logdir, extras={'quantized_top1': test_res[0]})

    del qe_model
    return test_res

def acts_quant_stats_collection(model, criterion, loggers, args, test_loader=None, save_to_file=False):
    msglogger.info('Collecting quantization calibration stats based on {:.1%} of test dataset'
                   .format(args.qe_calibration))
    if test_loader is None:
        tmp_args = copy.deepcopy(args)
        tmp_args.effective_test_size = tmp_args.qe_calibration
        # Batch size 256 causes out-of-memory errors on some models (due to extra space taken by
        # stats calculations). Limiting to 128 for now.
        # TODO: Come up with "smarter" limitation?
        tmp_args.batch_size = min(128, tmp_args.batch_size)
        test_loader = load_data(tmp_args, fixed_subset=True, load_train=False, load_val=False)
    test_fn = partial(test, test_loader=test_loader, criterion=criterion,
                      loggers=loggers, args=args, activations_collectors=None)
    with quantizing.get_nonparallel_clone_model(model) as cmodel:
        return collect_quant_stats(cmodel, test_fn, classes=None,
                                   inplace_runtime_check=True, disable_inplace_attrs=True,
                                   save_dir=msglogger.logdir if save_to_file else None)


def _config_determinism(args):
    if args.evaluate:
        args.deterministic = True
    
    # Configure some seed (in case we want to reproduce this experiment session)
    if args.seed is None:
        if args.deterministic:
            args.seed = 0
        else:
            args.seed = np.random.randint(1, 100000)

    if args.deterministic:
        quantizing.set_deterministic(args.seed) # For experiment reproducability
    else:
        quantizing.set_seed(args.seed)
        # Turn on CUDNN benchmark mode for best performance. This is usually "safe" for image
        # classification models, as the input sizes don't change during the run
        # See here: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
        cudnn.benchmark = True
    msglogger.info("Random seed: %d", args.seed)


def _config_compute_device(args):
    if args.cpu or not torch.cuda.is_available():
        # Set GPU index to -1 if using CPU
        args.device = 'cpu'
        args.gpus = -1
    else:
        args.device = 'cuda'
        if args.gpus is not None:
            try:
                args.gpus = [int(s) for s in args.gpus.split(',')]
            except ValueError:
                raise ValueError('ERROR: Argument --gpus must be a comma-separated list of integers only')
            available_gpus = torch.cuda.device_count()
            for dev_id in args.gpus:
                if dev_id >= available_gpus:
                    raise ValueError('ERROR: GPU device ID {0} requested, but only {1} devices available'
                                     .format(dev_id, available_gpus))
            # Set default device in case the first one on the list != 0
            torch.cuda.set_device(args.gpus[0])

def create_activation_stats_collectors(model, *phases):
    """Create objects that collect activation statistics.

    This is a utility function that creates two collectors:
    1. Fine-grade sparsity levels of the activations
    2. L1-magnitude of each of the activation channels

    Args:
        model - the model on which we want to collect statistics
        phases - the statistics collection phases: train, valid, and/or test

    WARNING! Enabling activation statsitics collection will significantly slow down training!
    """
    class missingdict(dict):
        """This is a little trick to prevent KeyError"""
        def __missing__(self, key):
            return None  # note, does *not* set self[key] - we don't want defaultdict's behavior

    genCollectors = lambda: missingdict({
        "sparsity_ofm":      SummaryActivationStatsCollector(model, "sparsity_ofm",
            lambda t: 100 * quantizing.utils.sparsity(t)),
        "l1_channels":   SummaryActivationStatsCollector(model, "l1_channels",
                                                         quantizing.utils.activation_channels_l1),
        "apoz_channels": SummaryActivationStatsCollector(model, "apoz_channels",
                                                         quantizing.utils.activation_channels_apoz),
        "mean_channels": SummaryActivationStatsCollector(model, "mean_channels",
                                                         quantizing.utils.activation_channels_means),
        "records":       RecordsActivationStatsCollector(model, classes=[torch.nn.Conv2d])
    })

    return {k: (genCollectors() if k in phases else missingdict())
            for k in ('train', 'valid', 'test')}

def init_classifier_compression_arg_parser_args(include_ptq_lapq_args=False):
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
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set')
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
    parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed the PRNG for CPU, CUDA, numpy, and Python')
    parser.add_argument('--gpus', metavar='DEV_ID', default=None,
                        help='Comma-separated list of GPU device IDs to be used '
                             '(default is to use all available devices)')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use CPU only. \n'
                        'Flag not set => uses GPUs according to the --gpus flag value.'
                        'Flag set => overrides the --gpus flag')
    parser.add_argument('--activation-stats', '--act-stats', nargs='+', metavar='PHASE', default=list(),
                        help='collect activation statistics on phases: train, valid, and/or test'
                        ' (WARNING: this slows down training)')
    parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                        help='configuration file for pruning the model (default is to use hard-coded schedule)')
    
    load_checkpoint_group = parser.add_argument_group('Resuming arguments')
    load_checkpoint_group_exc = load_checkpoint_group.add_mutually_exclusive_group()
    # TODO(barrh): args.deprecated_resume is deprecated since v0.3.1
    load_checkpoint_group_exc.add_argument('--resume', dest='deprecated_resume', default='', type=str,
                        metavar='PATH', help=argparse.SUPPRESS)
    load_checkpoint_group_exc.add_argument('--resume-from', dest='resumed_checkpoint_path', default='',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint. Use to resume paused training session.')
    load_checkpoint_group_exc.add_argument('--exp-load-weights-from', dest='load_model_path',
                        default='', type=str, metavar='PATH',
                        help='path to checkpoint to load weights from (excluding other fields) (experimental)')
    load_checkpoint_group.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    load_checkpoint_group.add_argument('--reset-optimizer', action='store_true',
                        help='Flag to override optimizer if resumed from checkpoint. This will reset epochs count.')
    
    str_to_quant_mode_map = OrderedDict([
        ('sym', LinearQuantMode.SYMMETRIC),
        ('sym_restr', LinearQuantMode.SYMMETRIC_RESTRICTED),
        ('asym_s', LinearQuantMode.ASYMMETRIC_SIGNED),
        ('asym_u', LinearQuantMode.ASYMMETRIC_UNSIGNED)
    ])

    str_to_clip_mode_map = OrderedDict([
        ('none', ClipMode.NONE), ('avg', ClipMode.AVG), ('n_std', ClipMode.N_STD),
        ('gauss', ClipMode.GAUSS), ('laplace', ClipMode.LAPLACE)
    ])

    def from_dict(val_str, d, optional):
        if not val_str and optional:
            return None
        try:
            return d[val_str]
        except KeyError:
            raise argparse.ArgumentTypeError('Must be one of {0} (received {1})'.format(list(d.keys()), val_str))

    linear_quant_mode_str = partial(from_dict, d=str_to_quant_mode_map, optional=False)
    linear_quant_mode_str_optional = partial(from_dict, d=str_to_quant_mode_map, optional=True)
    clip_mode_str = partial(from_dict, d=str_to_clip_mode_map, optional=False)

    group = parser.add_argument_group('Post-Training Quantization Arguments')
    group.add_argument('--quantize-eval', '--qe', action='store_true',
                       help='Apply linear quantization to model before evaluation. Applicable only if '
                            '--evaluate is also set')
    group.add_argument('--qe-mode', '--qem', type=linear_quant_mode_str, default='sym',
                       help='Default linear quantization mode (for weights and activations). '
                            'Choices: ' + ' | '.join(str_to_quant_mode_map.keys()))
    group.add_argument('--qe-mode-acts', '--qema', type=linear_quant_mode_str_optional, default=None,
                       help='Linear quantization mode for activations. Overrides --qe-mode`. '
                            'Choices: ' + ' | '.join(str_to_quant_mode_map.keys()))
    group.add_argument('--qe-mode-wts', '--qemw', type=linear_quant_mode_str_optional, default=None,
                       help='Linear quantization mode for Weights. Overrides --qe-mode`. '
                            'Choices: ' + ' | '.join(str_to_quant_mode_map.keys()))
    group.add_argument('--qe-bits-acts', '--qeba', type=int, default=8, metavar='NUM_BITS',
                       help='Number of bits for quantization of activations. Use 0 to not quantize activations. '
                            'Default value is 8')
    group.add_argument('--qe-bits-wts', '--qebw', type=int, default=8, metavar='NUM_BITS',
                       help='Number of bits for quantization of weights. Use 0 to not quantize weights. '
                            'Default value is 8')
    group.add_argument('--qe-bits-accum', type=int, default=32, metavar='NUM_BITS',
                       help='Number of bits for quantization of the accumulator')
    group.add_argument('--qe-clip-acts', '--qeca', type=clip_mode_str, default='none',
                       help='Activations clipping mode. Choices: ' + ' | '.join(str_to_clip_mode_map.keys()))
    group.add_argument('--qe-clip-n-stds', type=float,
                       help='When qe-clip-acts is set to \'n_std\', this is the number of standard deviations to use')
    group.add_argument('--qe-no-clip-layers', '--qencl', type=str, nargs='+', metavar='LAYER_NAME', default=[],
                       help='List of layer names for which not to clip activations. Applicable '
                            'only if --qe-clip-acts is not \'none\'')
    group.add_argument('--qe-no-quant-layers', '--qenql', type=str, nargs='+', metavar='LAYER_NAME', default=[],
                        help='List of layer names for which to skip quantization.')
    group.add_argument('--qe-per-channel', '--qepc', action='store_true',
                       help='Enable per-channel quantization of weights (per output channel)')
    group.add_argument('--qe-scale-approx-bits', '--qesab', type=int, metavar='NUM_BITS',
                       help='Enables scale factor approximation using integer multiply + bit shift, using '
                            'this number of bits the integer multiplier')
    group.add_argument('--qe-save-fp-weights', action='store_true',
                       help='Allow weights requantization.')
    group.add_argument('--qe-convert-pytorch', '--qept', action='store_true',
                       help='Convert the model to PyTorch native post-train quantization modules')
    group.add_argument('--qe-pytorch-backend', default='fbgemm', choices=['fbgemm', 'qnnpack'],
                       help='When --qe-convert-pytorch is set, specifies the PyTorch quantization backend to use')

    stats_group = group.add_mutually_exclusive_group()
    stats_group.add_argument('--qe-stats-file', type=str, metavar='PATH',
                             help='Path to YAML file with pre-made calibration stats')
    stats_group.add_argument('--qe-dynamic', action='store_true', help='Apply dynamic quantization')
    stats_group.add_argument('--qe-calibration', type=quantizing.utils.float_range_argparse_checker(exc_min=True),
                             metavar='PORTION_OF_TEST_SET', default=None,
                             help='Run the model in evaluation mode on the specified portion of the test dataset and '
                                  'collect statistics')
    stats_group.add_argument('--qe-config-file', type=str, metavar='PATH',
                             help='Path to YAML file containing configuration for PostTrainRLinearQuantizer '
                                  '(if present, all other --qe* arguments are ignored)')

    args = parser.parse_args()
    return args

def _init_logger(args, script_dir):
    global msglogger
    if script_dir is None or not hasattr(args, "output_dir") or args.output_dir is None:
        msglogger.logdir = None
        return None
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'),
                                         args.name, args.output_dir, args.verbose)

    # Log various details about the execution environment.  It is sometimes useful
    # to refer to past experiment executions and this information may be useful.
    apputils.log_execution_env_state(
        filter(None, [args.compress, args.qe_stats_file]),  # remove both None and empty strings
        msglogger.logdir)
    msglogger.debug("quantizing: %s", quantizing.__version__)
    return msglogger.logdir

def _init_learner(args):
    # Create the model
    model = create_models(args.dataset, args.arch)
    compression_scheduler = None

    # TODO(barrh): args.deprecated_resume is deprecated since v0.3.1
    if args.deprecated_resume:
        msglogger.warning('The "--resume" flag is deprecated. Please use "--resume-from=YOUR_PATH" instead.')
        if not args.reset_optimizer:
            msglogger.warning('If you wish to also reset the optimizer, call with: --reset-optimizer')
            args.reset_optimizer = True
        args.resumed_checkpoint_path = args.deprecated_resume

    optimizer = None
    start_epoch = 0
    if args.resumed_checkpoint_path:
        model, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(
            model, args.resumed_checkpoint_path, model_device=args.device)
    elif args.load_model_path:
        model = apputils.load_lean_checkpoint(model, args.load_model_path, model_device=args.device)
    if args.reset_optimizer:
        start_epoch = 0
        if optimizer is not None:
            optimizer = None
            msglogger.info('\nreset_optimizer flag set: Overriding resumed optimizer and resetting epoch count to 0')

    if optimizer is None and not args.evaluate:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        msglogger.debug('Optimizer Type: %s', type(optimizer))
        msglogger.debug('Optimizer Args: %s', optimizer.defaults)

    if args.compress:
        # The main use-case for this sample application is CNN compression. Compression
        # requires a compression schedule configuration file in YAML.
        compression_scheduler = quantizing.file_config(model, optimizer, args.compress, compression_scheduler,
            (start_epoch-1) if args.resumed_checkpoint_path else None)
        # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
        model.to(args.device)
    elif compression_scheduler is None:
        compression_scheduler = quantizing.CompressionScheduler(model)

    return model, compression_scheduler, optimizer, start_epoch, args.epochs


class QuantizeModel(object):
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self._infer_implicit_args(self.args)
        self.logdir = _init_logger(self.args, "logs")
        _config_determinism(self.args)
        _config_compute_device(self.args)
        
        # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
        # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.

        self.tflogger = TensorBoardLogger(msglogger.logdir)
        self.pylogger = PythonLogger(msglogger)
        (self.model, self.compression_scheduler, self.optimizer, 
             self.start_epoch, self.ending_epoch) = _init_learner(self.args)

        # Define loss function (criterion)
        self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        self.train_loader, self.val_loader, self.test_loader = (None, None, None)
        self.activations_collectors = create_activation_stats_collectors(
            self.model, *self.args.activation_stats)
        self.performance_tracker = apputils.SparsityAccuracyTracker(self.args.num_best_scores)
    
    @staticmethod
    def _infer_implicit_args(args):
        # Infer the dataset from the model name
        if not hasattr(args, 'dataset'):
            args.dataset = quantizing.apputils.classification_dataset_str_from_arch(args.arch)
        if not hasattr(args, "num_classes"):
            args.num_classes = quantizing.apputils.classification_num_classes(args.dataset)
        return args

    @staticmethod
    def mock_args():
        """Generate a Namespace based on default arguments"""
        return ClassifierCompressor._infer_implicit_args(
            init_classifier_compression_arg_parser().parse_args(['fictive_required_arg',]))

    def load_datasets(self):
        """Load the datasets"""
        if not all((self.train_loader, self.val_loader, self.test_loader)):
            self.train_loader, self.val_loader, self.test_loader = load_data(self.args)
        return self.data_loaders

    @property
    def data_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader
    
    def test(self):
        self.load_datasets()
        return test(self.test_loader, self.model, self.criterion,
                    self.pylogger, args=self.args)

    def quantize_evaluate(self):
        test_loader = load_data(self.args, load_train=False, load_val=False, load_test=True)
        evaluate_model(test_loader, self.model, self.criterion, self.pylogger,
                create_activation_stats_collectors(self.model, *self.args.activation_stats),
                self.args, scheduler=self.compression_scheduler)

def main():
    args = init_classifier_compression_arg_parser_args()
    model = QuantizeModel(args)
    model.quantize_evaluate()
    

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