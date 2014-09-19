#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from pylstm import Online, Monitor
from pylstm.error_functions import (
    CrossEntropyError, MultiClassCrossEntropyError, CTC, MeanSquaredError)
from pylstm.datasets.loader import get_dataset_specs
from pylstm.structure.layers import InputLayer, ForwardLayer


def setup_from_dataset(filename, variant=''):
    input_size, output_size, task_type = get_dataset_specs(filename, variant)
    if task_type == 'classification':
        if output_size == 1:
            output_act_func = 'sigmoid'
            error_func = CrossEntropyError
        else:
            output_act_func = 'softmax'
            error_func = MultiClassCrossEntropyError
    elif task_type == 'labeling':
        output_act_func = 'softmax'
        output_size += 1  # for the empty label
        error_func = CTC
    else:  # task_type == 'regression'
        output_act_func = 'linear'
        error_func = MeanSquaredError

    input_layer = InputLayer(input_size)
    output_layer = ForwardLayer(output_size, act_func=output_act_func,
                                name='OutputLayer')

    return input_layer, output_layer, error_func


def evaluate(net, input_data, targets, error_function):
    errors = []
    for x, t in Online(input_data, targets, shuffle=False)(verbose=True):
        y = net.forward_pass(x)
        error, _ = error_function(y, t)
        errors.append(error)

    return error_function.aggregate(errors)


class InfoUpdater(Monitor):
    def __init__(self, run, name=None):
        super(InfoUpdater, self).__init__(name, 'epoch', 1)
        self.run = run
        self.__name__ = self.__class__.__name__ if name is None else name

    def __call__(self, epoch, net, stepper, logs):
        info = self.run.info
        info['epochs_needed'] = epoch
        info['monitor'] = logs
        if 'nr_parameters' not in info:
            info['nr_parameters'] = net.get_param_size()