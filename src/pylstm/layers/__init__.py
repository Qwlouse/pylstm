#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from .construction_layer import create_construction_layer
from .construction_layer import InvalidArchitectureError

# python layers
InputLayer = create_construction_layer('InputLayer')
OutputLayer = create_construction_layer('OutputLayer')


def _create_construction_layers_for_python_layers():
    """
    This method will create a ConstructionLayer object for every '...Layer' in
    python_layers and add it to this modules namespace.
    """
    import python_layers
    PYTHON_LAYERS = {n: v for n, v in python_layers.__dict__.items()
                     if not n.startswith('_') and n.endswith('Layer')}
    import sys
    module = sys.modules[__name__]
    for name, value in PYTHON_LAYERS.iteritems():
        setattr(module, name, create_construction_layer(value))

_create_construction_layers_for_python_layers()


# c++ layers
LstmLayer = create_construction_layer("LstmLayer")
Lstm97Layer = create_construction_layer("Lstm97Layer")
RnnLayer = create_construction_layer("RnnLayer")
ArnnLayer = create_construction_layer("ArnnLayer")
MrnnLayer = create_construction_layer("MrnnLayer")
RegularLayer = create_construction_layer("RegularLayer")
ReverseLayer = create_construction_layer("ReverseLayer")