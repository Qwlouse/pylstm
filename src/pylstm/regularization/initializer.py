#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np


class Gaussian(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, layer_name, view_name,  shape, seed=None):
        rnd = np.random.RandomState(seed)
        size = reduce(np.multiply, shape)
        return rnd.randn(size).reshape(*shape) * self.std + self.mean


class Uniform(object):
    def __init__(self, low=-0.1, high=0.1):
        self.low = low
        self.high = high

    def __call__(self, layer_name, view_name, shape, seed=None):
        rnd = np.random.RandomState(seed)
        size = reduce(np.multiply, shape)
        v = ((self.high - self.low) * rnd.rand(size).reshape(*shape)) + self.low
        return v


class CopyFromNetwork(object):
    def __init__(self, net, default=0):
        self.net = net
        self.default = default

    def __call__(self, layer_name, view_name, shape, seed=None):
        if layer_name in self.net.layers:
            layer_view = self.net.get_param_view_for(layer_name)
            if view_name in layer_view:
                view = layer_view[view_name]
                if view.shape == shape:
                    return view
                else:
                    print('CopyFromNetwork: Shape mismatch %s != %s in view %s '
                          'of %s.' % (view.shape, shape, view_name, layer_name))
            else:
                print('CopyFromNetwork: View %s not found in layer %s.' %
                      (view_name, layer_name))
        else:
            print('CopyFromNetwork: Layer %s not found.' % layer_name)

        if callable(self.default):
            return self.default(layer_name, view_name, shape, seed)
        else:
            return self.default


class InSparse(object):
    def __init__(self, init, connections=15):
        self.init = init
        self.connections = connections

    def __call__(self, layer_name, view_name,  shape, seed=None):
        res = self.init(layer_name, view_name,  shape, seed)
        if shape[1] == 1:  # Just one input: probably bias => ignore
            return res
        assert shape[0] == 1  # weights don't have a time axis
        assert shape[1] >= self.connections
        rnd = np.random.RandomState(seed)
        M = np.zeros(shape)
        M[0, :self.connections, :] = 1.
        for i in range(shape[2]):
            rnd.shuffle(M[0, :, i])
        return res * M


class OutSparse(object):
    def __init__(self, init, connections=15):
        self.init = init
        self.connections = connections

    def __call__(self, layer_name, view_name,  shape, seed=None):
        res = self.init(layer_name, view_name,  shape, seed)
        if shape[1] == 1:  # Just one input: probably bias => ignore
            return res
        assert shape[0] == 1  # weights don't have a time axis
        assert shape[2] >= self.connections
        rnd = np.random.RandomState(seed)
        M = np.zeros(shape)
        M[0, :, :self.connections] = 1.
        for i in range(shape[1]):
            rnd.shuffle(M[0, i, :])
        return res * M






