#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from ..wrapper import create_layer as create_c_layer
import python_layers

__all__ = ['InvalidArchitectureError', 'ConstructionLayer',
           'create_construction_layer']


class InvalidArchitectureError(Exception):
    pass


class ConstructionLayer(object):
    def __init__(self, layer_type, out_size, name=None, **layer_kwargs):
        self.out_size = out_size
        self.name = name
        self.targets = []
        self.sources = []
        self.layer_type = layer_type
        self.layer_kwargs = layer_kwargs
        self.depth = None
        self.traversing = False

    def instantiate(self):
        return _create_layer(self.layer_type, self.get_input_size(),
                             self.out_size, self.layer_kwargs)

    def get_input_size(self):
        return sum(s.out_size for s in self.sources)

    def get_depth(self):
        if self.depth is None:
            self.depth = float('inf')  # marker for "get_depth" in progress
            self.depth = max(s.get_depth() for s in self.sources) + 1
        return self.depth

    def traverse_targets_tree(self):
        if self.traversing:
            raise InvalidArchitectureError("Circle in Network")
        self.traversing = True
        yield self
        for target in self.targets:
            for t in target.traverse_targets_tree():
                yield t
        self.traversing = False

    def _add_source(self, other):
        self.sources.append(other)

    def __rshift__(self, other):
        self.targets.append(other)
        other._add_source(self)
        return other

    def get_name(self):
        if self.name:
            return self.name
        elif isinstance(self.layer_type, basestring):
            return self.layer_type
        else:
            return self.layer_type.__name__

    def __repr__(self):
        return "<ConstructionLayer: %s>" % (self.get_name())


def create_construction_layer(layer_type):
    def constructor(*args, **kwargs):
        return ConstructionLayer(layer_type, *args, **kwargs)

    return constructor


def _create_layer(name, input_size, output_size, kwargs):
    if name in python_layers.__dict__:
        LayerType = python_layers.__dict__[name]
        return LayerType(input_size, output_size, **kwargs)
    else:
        return create_c_layer(name, input_size, output_size, **kwargs)

