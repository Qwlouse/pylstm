#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from .construction_layer import create_construction_layer
from .construction_layer import InvalidArchitectureError

# python layers
Input = create_construction_layer('Input')
Output = create_construction_layer('Output')


def _create_construction_layers_for_python_layers():
    """
    This method will create a ConstructionLayer object for every '...Layer' in
    python_layers and add it to this modules namespace.
    """
    import python_layers
    PYTHON_LAYERS = {n[:-5]: v for n, v in python_layers.__dict__.items()
                     if not n.startswith('_') and n.endswith('Layer')}
    import sys
    module = sys.modules[__name__]
    for name, value in PYTHON_LAYERS.iteritems():
        setattr(module, name, create_construction_layer(value))

_create_construction_layers_for_python_layers()


# c++ layers
LSTM = create_construction_layer("LSTM")
LSTM97 = create_construction_layer("LSTM97")
RNN = create_construction_layer("RNN")
ARNN = create_construction_layer("ARNN")
MRNN = create_construction_layer("MRNN")
Regular = create_construction_layer("Regular")
Reverse = create_construction_layer("Reverse")