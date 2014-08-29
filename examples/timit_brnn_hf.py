#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from pylstm import *

hidden_size = 100
input_layer = InputLayer(39)
output_layer = HfFinalLayer(61, act_func='softmax', name='OutputLayer')
input_layer >> RnnLayer(hidden_size, name='ForwardRNN', act_func='tanh') >> output_layer
input_layer >> ReverseLayer() >> RnnLayer(hidden_size, name='BackwardRNN', act_func='tanh') >> ReverseLayer() >> output_layer
net = build_net(output_layer)
net.error_func = MultiClassCrossEntropyError

net.initialize(default=Gaussian(std=.25),
               ForwardRNN={'HR': SparseInputs(Gaussian(std=.25), connections=15),
                           'HX': SparseOutputs(Gaussian(std=1), connections=15),
                           },
               BackwardRNN={'HR': SparseInputs(Gaussian(std=.25), connections=15),
                            'HX': SparseOutputs(Gaussian(std=1), connections=15),
                            },
               OutputLayer={'HX': SparseInputs(Gaussian(std=.25), connections=15)})


def print_lambda(epoch, stepper, **_):
    print('lambda:', stepper.lambda_)


ds = load_dataset('/home/greff/ds/timit', 'rtimit')
X, T, M = ds['train']
T_b = binarize_array(T, range(61))
ds['train'] = X, T_b, M
X, T, M = ds['val']
T_b = binarize_array(T, range(61))
ds['val'] = X, T_b, M
X, T, M = ds['test']
T_b = binarize_array(T, range(61))
ds['test'] = X, T_b, M


tr = Trainer(net, CgStep(minibatch_size=100))

tr.add_stopper(ValidationErrorRises(10))
tr.add_stopper(MaxEpochsSeen(120))

tr.add_monitor(PrintError())
tr.add_monitor(MonitorClassificationError(
    Online(*ds['test'], shuffle=False, verbose=False),
    name='testError'))
tr.add_monitor(print_lambda)
tr.add_monitor(SaveBestWeights('timit_brnn_nf.npy'))

tr.train(Undivided(*ds['train'], shuffle=False),
         Undivided(*ds['val'], shuffle=False))