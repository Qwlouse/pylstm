#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from .construction_layer import create_ConstructionLayer
from .construction_layer import InvalidArchitectureError

# python layers
InputLayer = create_ConstructionLayer('BaseLayer')
OutputLayer = create_ConstructionLayer('BaseLayer')


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
        setattr(module, name, create_ConstructionLayer(value))

_create_construction_layers_for_python_layers()


# c++ layers
LstmLayer = create_ConstructionLayer("LstmLayer")
Lstm97Layer = create_ConstructionLayer("Lstm97Layer")
RnnLayer = create_ConstructionLayer("RnnLayer")
ArnnLayer = create_ConstructionLayer("ArnnLayer")
MrnnLayer = create_ConstructionLayer("MrnnLayer")
RegularLayer = create_ConstructionLayer("RegularLayer")
ReverseLayer = create_ConstructionLayer("ReverseLayer")