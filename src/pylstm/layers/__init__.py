#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from .construction_layer import create_ConstructionLayer
from .construction_layer import InvalidArchitectureError

# python layers
DummyLayer = create_ConstructionLayer("DummyLayer")
CopyLayer = create_ConstructionLayer("CopyLayer")

# c++ layers
LstmLayer = create_ConstructionLayer("LstmLayer")
Lstm97Layer = create_ConstructionLayer("Lstm97Layer")
RnnLayer = create_ConstructionLayer("RnnLayer")
ArnnLayer = create_ConstructionLayer("ArnnLayer")
MrnnLayer = create_ConstructionLayer("MrnnLayer")
RegularLayer = create_ConstructionLayer("RegularLayer")
ReverseLayer = create_ConstructionLayer("ReverseLayer")