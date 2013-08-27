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


def assert_wellformed(net, initializers):
    all_layers = net.layers.keys()[1:]
    # assert only valid layers are specified
    for layer_name in initializers:
        assert layer_name == 'default' or layer_name in all_layers, \
            "Unknown Layer '%s'.\nPossible layers are: %s" % \
            (layer_name, ", ".join(all_layers))

    # assert only valid views are specified
    for layer_name, layer_initializers in initializers.items():
        if layer_name == 'default':
            continue
        param_view = net.get_param_view_for(layer_name)
        if isinstance(layer_initializers, dict):
            for view_name in layer_initializers:
                assert view_name == 'default' or view_name in param_view, \
                    "Unknown view '%s' for '%s'.\nPossible views are: %s"\
                    % (view_name, layer_name, ", ".join(param_view.keys()))


def _get_default_aware(values, layer_name, view_name):
    """
    This function retrieves values from an incomplete view dictionary that might
    make use of 'default'. These are used for initialize, set_regularizers, and
    set_constraints.
    """
    if not isinstance(values, dict):
        return values

    if layer_name in values:
        layer_values = values[layer_name]
        if not isinstance(layer_values, dict):
            return layer_values

        if view_name in layer_values:
            return layer_values[view_name]
        elif 'default' in layer_values:
            return layer_values['default']

    if 'default' in values:
        return values['default']

    return 0


def initialize(net, init_dict=(), seed=None, **kwargs):
    """
    This will initialize the network as specified and in the future(TM)
    return a serialized initialization specification.
    You can specify a seed to make the initialization reproducible.

    Example Usage:
        # you can set initializers in two equivalent ways:
        1) by passing a dictionary:
        >> initialize(net, {'RegularLayer': Uniform(), 'LstmLayer': Gaussian()})

        2) by using keyword arguments:
        >> initialize(net, RegularLayer=Uniform(), LstmLayer=Uniform())

        (you should not combine the two. If you do, however, then the keyword
         arguments take precedence)

        An initializer can either be a callable that takes
        (layer_name, view_name,  shape, seed) or something that converts to a
        numpy array. So for example:
        >> initialize(net,
                      LstmLayer=1,
                      RnnLayer=[1, 1, 1, 1, 1],
                      ForwardLayer=lambda ln, vn, s, seed: 1)

        You can specify a default initializer. If you don't a default=0 is
        assumed.
        >> initialize(net, default=Gaussian())

        To target only some of the weights of a layer can pass in a dict:
        >> initialize(net, LstmLayer={'HX': Gaussian(std=0.1),
                                      'IX': 0,
                                      'default': Uniform(-.1, .1)})
        The use of 'default' targets all unspecified views of the layer.
    """
    initializers = dict(init_dict)
    initializers.update(kwargs)
    assert_wellformed(net, initializers)

    rnd = np.random.RandomState(seed)

    for layer_name, layer in net.layers.items()[1:]:
        views = net.get_param_view_for(layer_name)
        for view_name, view in views.items():
            view_initializer = _get_default_aware(initializers,
                                                  layer_name,
                                                  view_name)
            if callable(view_initializer):
                view[:] = view_initializer(layer_name,
                                           view_name,
                                           view.shape,
                                           seed=rnd.randint(0, 1e9))
            else:
                view[:] = np.array(view_initializer)

    # TODO: implement serialization of initializer
    # it might be difficult to serialize custom initializers or lambda functions

    return 'not_implemented_yet'

