#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np


class InitializationFailedError(Exception):
    pass


class Gaussian(object):
    """
    Initializes the weights randomly according to a normal distribution of
    given mean and standard deviation.
    """
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, layer_name, view_name,  shape, seed=None):
        rnd = np.random.RandomState(seed)
        size = reduce(np.multiply, shape)
        return rnd.randn(size).reshape(*shape) * self.std + self.mean


class Uniform(object):
    """
    Initializes the weights randomlay according to a uniform distribution over
    the interval [low; high].
    """
    def __init__(self, low=-0.1, high=0.1):
        self.low = low
        self.high = high

    def __call__(self, layer_name, view_name, shape, seed=None):
        rnd = np.random.RandomState(seed)
        size = reduce(np.multiply, shape)
        v = ((self.high - self.low) * rnd.rand(size).reshape(*shape)) + self.low
        return v


class CopyFromNetwork(object):
    """
    Initializes the weights by copying them from a target network.

    By default it first tries to find a layer with the same name in the target
    network (can be changed with layer_name parameter). If no matching layer is
    found in the target network the InitializationFailedError is raised.

    Then it tries to find a parameter view with the same name in the matching
    layer (can be changed with view_name parameter). If no matching view is
    found in the matching layer, then the on_missing_view initializer is
    evaluated instead. If none is given the InitializationFailedError is raised.

    If the shape of the matching view does not match the on_shape_mismatch
    initializer is evaluated instead. If none is given the
    InitializationFailedError is raised.

    Finally, if everything was alright it copies the weights.

    Example usage:
    >> net.initialize(default=Uniform(),
                      LstmLayer_1=CopyFromNetwork(onet, layer_name='LstmLayer'))
    This will copy the weights of 'LstmLayer' from onet into 'LstmLayer_1' of
    net, and initialize the rest with Uniform distribution.
    """
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
                    return _evaluate_initializer(self.on_shape_mismatch,
                                                layer_name, view_name, shape,
                                                seed)
                else:
                    raise InitializationFailedError('Shape mismatch %s != %s '
                                                    'in view %s of %s.' %
                                                    (view.shape, shape,
                                                     view_name, layer_name))
            elif self.on_missing_view is not None:
                return _evaluate_initializer(self.on_missing_view,
                                            layer_name, view_name, shape,
                                            seed)
            else:
                raise InitializationFailedError('View %s not found in layer %s.'
                                                % (view_name, layer_name))
        else:
            raise InitializationFailedError('Layer %s not found.' % layer_name)


class SparseInputs(object):
    """
    Makes sure every neuron only gets activation from a certain number of input
    neurons and the rest of the weights are 0.
    The connections are initialized by evaluating the passed in init.

    Example usage:
    >> net = build_net(InputLayer(20) >> ForwardLayer(5))
    >> net.initialize(ForwardLayer=SparseInputs(Gaussian(), connections=10))
    """
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


class SparseOutputs(object):
    """
    Makes sure every neuron is propagating its activation only to a certain
    number of output neurons, and the rest of the weights are 0.
    The connections are initialized by evaluating the passed in init.

    Example usage:
    >> net = build_net(InputLayer(5) >> ForwardLayer(20))
    >> net.initialize(ForwardLayer=SparseOutputs(Gaussian(), connections=10))
    """

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


def _evaluate_initializer(initializer, layer_name, view_name, shape, seed):
    if callable(initializer):
        return initializer(layer_name, view_name, shape, seed)
    else:
        return np.array(initializer)


