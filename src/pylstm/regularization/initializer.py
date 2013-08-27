#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np


class InitializationFailedError(Exception):
    pass


def evaluate_initializer(initializer, layer_name, view_name, shape, seed):
    if callable(initializer):
        return initializer(layer_name, view_name, shape, seed)
    else:
        return np.array(initializer)


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
    def __init__(self, net, layer_name=None, view_name=None,
                 on_missing_view=None, on_shape_mismatch=None):
        self.net = net
        self.layer_name = layer_name
        self.view_name = view_name
        self.on_missing_view = on_missing_view
        self.on_shape_mismatch = on_shape_mismatch

    def __call__(self, layer_name, view_name, shape, seed=None):
        layer_name = layer_name if self.layer_name is None else self.layer_name
        view_name = view_name if self.view_name is None else self.view_name

        if layer_name in self.net.layers:
            layer_view = self.net.get_param_view_for(layer_name)
            if view_name in layer_view:
                view = layer_view[view_name]
                if view.shape == shape:
                    return view
                elif self.on_shape_mismatch is not None:
                    return evaluate_initializer(self.on_shape_mismatch,
                                                layer_name, view_name, shape,
                                                seed)
                else:
                    raise InitializationFailedError('Shape mismatch %s != %s '
                                                    'in view %s of %s.' %
                                                    (view.shape, shape,
                                                     view_name, layer_name))
            elif self.on_missing_view is not None:
                return evaluate_initializer(self.on_missing_view,
                                            layer_name, view_name, shape,
                                            seed)
            else:
                raise InitializationFailedError('View %s not found in layer %s.'
                                                % (view_name, layer_name))
        else:
            raise InitializationFailedError('Layer %s not found.' % layer_name)


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






