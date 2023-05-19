#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""This package contains ImageNet and CIFAR image classification models for pytorch"""

import copy
from functools import partial
import torch
import torchvision.models as torch_models
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.ops.misc import FrozenBatchNorm2d
import torch.nn as nn
from . import cifar10 as cifar10_models
import pretrainedmodels

from quantizing.utils import set_model_input_shape_attr, model_setattr
from quantizing.modules import Mean, EltwiseAdd

import logging
msglogger = logging.getLogger()

SUPPORTED_DATASETS = ('imagenet', 'cifar10', 'mnist')

# ResNet special treatment: we have our own version of ResNet, so we need to over-ride
# TorchVision's version.
RESNET_SYMS = ('ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
               'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2')

TORCHVISION_MODEL_NAMES = sorted(
                            name for name in torch_models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(torch_models.__dict__[name]))

CIFAR10_MODEL_NAMES = sorted(name for name in cifar10_models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(cifar10_models.__dict__[name]))


_model_extensions = {}


def create_models(dataset, arch):
    dataset = dataset.lower()
    model = None
    cadene = False
    if dataset == 'cifar10':
        model = _create_cifar10_model(arch)
    else:
        raise ValueError('Could not recognize dataset {} and arch {} pair'.format(dataset, arch))
    if torch.cuda.is_available():
        print("gpu")
        device = 'cuda'
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.is_parallel = True
    else:
        print("cpu")
        device = 'cpu'
        model.is_parallel = False
    set_model_input_shape_attr(model, dataset=dataset)
    model.arch = arch
    model.dataset = dataset
    return model.to(device)



def _create_cifar10_model(arch):
    try:
        model = cifar10_models.__dict__[arch]()
    except KeyError:
        raise ValueError("Model {} is not supported for dataset CIFAR10".format(arch))
    return model

def _set_model_input_shape_attr(model, arch, dataset, pretrained, cadene):
    if cadene and pretrained:
        # When using pre-trained weights, Cadene models already have an input size attribute
        # We add the batch dimension to it
        input_size = model.module.input_size if isinstance(model, torch.nn.DataParallel) else model.input_size
        shape = tuple([1] + input_size)
        set_model_input_shape_attr(model, input_shape=shape)
    elif arch == 'inception_v3':
        set_model_input_shape_attr(model, input_shape=(1, 3, 299, 299))
    else:
        set_model_input_shape_attr(model, dataset=dataset)


def register_user_model(arch, dataset, model):
    """A simple mechanism to support models that are not part of distiller.models"""
    _model_extensions[(arch, dataset)] = model


def _is_registered_extension(arch, dataset, pretrained):
    try:
        return _model_extensions[(arch, dataset)] is not None
    except KeyError:
        return False


def _create_extension_model(arch, dataset):
    return _model_extensions[(arch, dataset)]()