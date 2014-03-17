#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from pylstm.randomness import Seedable, SEEDABLE_MEMBERS


class InitializationError(Exception):
    pass


def create_initializer_from_description(description):
    """
    Turn an initialization-description into a initialization dictionary, that
    can be used with Network.initialize.

    :param description: initialization-description
    :type description: dict

    :return: initialization-dictionary
    :rtype: dict
    """
    if isinstance(description, dict):
        if '$type' in description:
            name = description['$type']
            for initializer in Initializer.__subclasses__():
                if initializer.__name__ == name:
                    instance = initializer.__new__(initializer)
                    instance.__init_from_description__(description)
                    return instance
            raise InitializationError('Initializer "%s" not found!' % name)
        else:
            return {k: create_initializer_from_description(v)
                    for k, v in description.items()}
    elif isinstance(description, (list, int, long, float)):
        return description
    else:
        raise InitializationError('illegal description type "%s"' %
                                  type(description))


def get_initializer_description(initializer):
    """
    Turn a initialization-dictionary as used in the Network.initialize method
    into a description dictionary. This description is json serializable.

    :param initializer: initialization-dictionary
    :type initializer: dict
    :return: description
    :rtype: dict
    """
    if isinstance(initializer, Initializer):
        return initializer.__get_description__()
    elif isinstance(initializer, dict):
        return {k: get_initializer_description(v)
                for k, v in initializer.items()}
    elif isinstance(initializer, np.ndarray):
        return initializer.tolist()
    else:
        return initializer


class Initializer(Seedable):
    """
    Base Class for all initializers. It inherits from Seedable, so every
    sub-class has access to self.rnd, and it provides basic methods for
    converting from and to a description.
    """
    def __get_description__(self):
        """
        Returns a description of the initializer. That is a dictionary
        containing the name of the class as '$type' and all members of the
        class. It does not contain the members of Seedable though.

        If a sub-class of Initializer contains
        non-simple (numerical, string, list[numerical]) fields it has to
        override this method to specify how they should be described.

        :rtype: dict
        """
        description = {k: v for k, v in self.__dict__.items()
                       if k not in SEEDABLE_MEMBERS}
        description['$type'] = self.__class__.__name__
        return description

    def __init_from_description__(self, description):
        """
        Initializes an initializer from a given description.

        Again if a sub-class of Initializer contains
        non-simple (numerical, string, list[numerical]) fields it has to
        override this method to specify how they should be initialized from
        their description.

        :param description: description of this Initializer object
        :type description: dict
        """
        assert self.__class__.__name__ == description['$type']
        Seedable.__init__(self)
        self.__dict__.update({k: v for k, v in description.items()
                              if k != '$type'})

    def __call__(self, layer_name, view_name,  shape):
        raise NotImplementedError()


class Gaussian(Initializer):
    """
    Initializes the weights randomly according to a normal distribution of
    given mean and standard deviation.
    """

    def __init__(self, mean=0.0, std=1.0):
        super(Gaussian, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, layer_name, view_name,  shape):
        size = reduce(np.multiply, shape)
        return self.rnd.randn(size).reshape(*shape) * self.std + self.mean


class Uniform(Initializer):
    """
    Initializes the weights randomly according to a uniform distribution over
    the interval [low; high].
    """

    def __init__(self, low=-0.1, high=0.1):
        super(Uniform, self).__init__()
        self.low = low
        self.high = high

    def __call__(self, layer_name, view_name, shape):
        size = reduce(np.multiply, shape)
        v = ((self.high - self.low) * self.rnd.rand(size).reshape(*shape)) +\
            self.low
        return v


class DenseSqrtFanIn(Initializer):
    """
    Initializes the weights randomly according to a uniform distribution over
    the interval [-1/sqrt(n), 1/sqrt(n)] where n is the number of inputs to each
    neuron.
    """

    def __init__(self, scale=1.0):
        super(DenseSqrtFanIn, self).__init__()
        self.scale = scale

    def __call__(self, layer_name, view_name,  shape):
        size = reduce(np.multiply, shape)
        return self.scale * (2 * self.rnd.rand(size).reshape(*shape) - 1) /\
            np.sqrt(shape[1])


class DenseSqrtFanInOut(Initializer):
    """
    Initializes the weights randomly according to a uniform distribution over
    the interval [-1/sqrt(n1+n2), 1/sqrt(n1+n2)] where n1 is the number of
    inputs to each neuron and n2 is the number of neurons in the current layer.
    Use scaling = 4*sqrt(6) (used by default) for sigmoid units and sqrt(6) for
    tanh units.
    """

    def __init__(self, scale=4 * np.sqrt(6)):
        super(DenseSqrtFanInOut, self).__init__()
        self.scale = scale

    def __call__(self, layer_name, view_name,  shape):
        size = reduce(np.multiply, shape)
        return self.scale * (2 * self.rnd.rand(size).reshape(*shape) - 1) /\
            np.sqrt(shape[1] + shape[2])


class CopyFromNetwork(Initializer):
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
        super(CopyFromNetwork, self).__init__()
        self.net = net
        self.layer_name = layer_name
        self.view_name = view_name
        self.on_missing_view = on_missing_view
        self.on_shape_mismatch = on_shape_mismatch

    def __call__(self, layer_name, view_name, shape):
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
                                                 layer_name, view_name, shape)
                else:
                    raise InitializationError('Shape mismatch %s != %s in view'
                                              ' %s of %s.' % (view.shape,
                                                              shape, view_name,
                                                              layer_name))
            elif self.on_missing_view is not None:
                return _evaluate_initializer(self.on_missing_view,
                                             layer_name, view_name, shape)
            else:
                raise InitializationError('View %s not found in layer %s.' %
                                          (view_name, layer_name))
        else:
            raise InitializationError('Layer %s not found.' % layer_name)

    def __get_description__(self):
        raise NotImplementedError('CopyFromNetwork can not be turned into a '
                                  'description!')


class SparseInputs(Initializer):
    """
    Makes sure every neuron only gets activation from a certain number of input
    neurons and the rest of the weights are 0.
    The connections are initialized by evaluating the passed in init.

    Example usage:
    >> net = build_net(InputLayer(20) >> ForwardLayer(5))
    >> net.initialize(ForwardLayer=SparseInputs(Gaussian(), connections=10))
    """

    def __init__(self, init, connections=15):
        super(SparseInputs, self).__init__()
        self.init = init
        self.connections = connections

    def __call__(self, layer_name, view_name,  shape):
        res = self.init(layer_name, view_name,  shape)
        if shape[1] == 1:  # Just one input: probably bias => ignore
            return res
        assert shape[0] == 1  # weights don't have a time axis
        assert shape[1] >= self.connections

        connection_mask = np.zeros(shape)
        connection_mask[0, :self.connections, :] = 1.
        for i in range(shape[2]):
            self.rnd.shuffle(connection_mask[0, :, i])
        return res * connection_mask

    def __get_description__(self):
        return {
            '$type': self.__class__.__name__,
            'init': self.init.__get_description__(),
            'connections': self.connections
        }

    def __init_from_description__(self, description):
        assert self.__class__.__name__ == description['$type']
        self.init = create_initializer_from_description(description['init'])
        self.connections = description['connections']


class SparseOutputs(Initializer):
    """
    Makes sure every neuron is propagating its activation only to a certain
    number of output neurons, and the rest of the weights are 0.
    The connections are initialized by evaluating the passed in init.

    Example usage:
    >> net = build_net(InputLayer(5) >> ForwardLayer(20))
    >> net.initialize(ForwardLayer=SparseOutputs(Gaussian(), connections=10))
    """

    def __init__(self, init, connections=15):
        super(SparseOutputs, self).__init__()
        self.init = init
        self.connections = connections

    def __call__(self, layer_name, view_name,  shape):
        res = self.init(layer_name, view_name,  shape)
        if shape[1] == 1:  # Just one input: probably bias => ignore
            return res
        assert shape[0] == 1  # weights don't have a time axis
        assert shape[2] >= self.connections
        connection_mask = np.zeros(shape)
        connection_mask[0, :, :self.connections] = 1.
        for i in range(shape[1]):
            self.rnd.shuffle(connection_mask[0, i, :])
        return res * connection_mask

    def __get_description__(self):
        return {
            '$type': self.__class__.__name__,
            'init': self.init.__get_description(),
            'connections': self.connections
        }

    def __init_from_description__(self, description):
        assert self.__class__.__name__ == description['$type']
        self.init = create_initializer_from_description(description['init'])
        self.connections = description['connections']


class EchoState(Initializer):
    """
    Classic echo state initialization. Creates a matrix with a fixed spectral
    radius (default=1.25). Spectral radius should be < 1 to satisfy ES-property.
    Only works for square matrices.

    Example usage:
    >> net = build_net(InputLayer(5) >> RnnLayer(20, act_func='tanh'))
    >> net.initialize(default=Gaussian(), RnnLayer={'HR': EchoState(0.77)})
    """

    def __init__(self, spectral_radius=0.9):
        super(EchoState, self).__init__()
        self.spectral_radius = spectral_radius

    def __call__(self, layer_name, view_name,  shape):
        assert shape[0] == 1, "Shape should be 2D but was: %s" % str(shape)
        assert shape[1] == shape[2], \
            "Matrix should be square but was: %s" % str(shape)
        n = shape[1]
        weights = self.rnd.uniform(-0.5, 0.5, size=(n, n))
        # normalizing and setting spectral radius (correct, slow):
        rho_weights = max(abs(np.linalg.eig(weights)[0]))
        return weights.reshape(1, n, n) * (self.spectral_radius / rho_weights)


def _evaluate_initializer(initializer, layer_name, view_name, shape):
    if isinstance(initializer, Initializer):
        return initializer(layer_name, view_name, shape)
    else:
        return np.array(initializer)
