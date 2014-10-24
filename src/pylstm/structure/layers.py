#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from .construction_layer import create_construction_layer


# python layers
InputLayer = create_construction_layer('InputLayer')
NoOpLayer = create_construction_layer('NoOpLayer')
DeltaScalingLayer = create_construction_layer('DeltaScalingLayer')

# c++ layers
LstmLayer = create_construction_layer("LstmLayer")
StaticLstmLayer = create_construction_layer("StaticLstmLayer")
Lstm97Layer = create_construction_layer("Lstm97Layer")
RnnLayer = create_construction_layer("RnnLayer")
ClockworkLayer = create_construction_layer("ClockworkLayer")
MrnnLayer = create_construction_layer("MrnnLayer")
ForwardLayer = create_construction_layer("ForwardLayer")
ReverseLayer = create_construction_layer("ReverseLayer")
DropoutLayer = create_construction_layer("DropoutLayer")
HfFinalLayer = create_construction_layer("HfFinalLayer")
LWTALayer = create_construction_layer("LWTALayer")