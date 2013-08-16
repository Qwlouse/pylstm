#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from .construction_layer import create_construction_layer
from .construction_layer import InvalidArchitectureError

# python layers
InputLayer = create_construction_layer('InputLayer')
NoOpLayer = create_construction_layer('NoOpLayer')

# c++ layers
LstmLayer = create_construction_layer("LstmLayer")
Lstm97Layer = create_construction_layer("Lstm97Layer")
RnnLayer = create_construction_layer("RnnLayer")
ArnnLayer = create_construction_layer("ArnnLayer")
MrnnLayer = create_construction_layer("MrnnLayer")
ForwardLayer = create_construction_layer("ForwardLayer")
ReverseLayer = create_construction_layer("ReverseLayer")