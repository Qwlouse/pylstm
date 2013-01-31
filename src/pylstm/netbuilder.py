#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from buffer_manager import BufferManager
from layers import DummyLayer, InvalidArchitectureError
from network import Network
import wrapper

class NetworkBuilder():
    def __init__(self):
        self.input_layer = None
        self.output = DummyLayer(0, "Output")

    def input(self, size=None):
        if size :
            self.input_layer = DummyLayer(size, "Input")
        return self.input_layer

    def get_sorted_layers(self):
        if not self.input_layer:
            raise InvalidArchitectureError("Empty")
        # gather all the layers
        layers = set(self.input_layer.traverse_targets_tree())
        # sort them by depth
        self.input_layer.depth = 0
        return sorted(layers, key=lambda l: l.get_depth())

    def get_forward_closure(self, layer):
        """
        for a given layer return two sets of layers such that:
          * the given layer is in the first set
          * the second set contains all the target layers of the first set
          * the first set contains all the source layers of the second set
        """
        lset = {layer}
        rset = set(layer.targets)
        growing = True
        while growing:
            growing = False
            new_lset = {s for l in rset for s in l.sources}
            new_rset = {t for l in lset for t in l.targets}
            if len(new_lset) > len(lset) or\
               len(new_rset) > len(rset)    :
                growing = True
                lset = new_lset
                rset = new_rset
        return lset, rset

    def create_buffer(self, size):
        return wrapper.BufferView(1, 1, size)

    def get_named_layers(self):
        # instantiate all the layers with names
        cLayers = self.get_sorted_layers()
        assert cLayers[0] is self.input_layer
        assert cLayers[-1] is self.output
        layers = OrderedDict()
        for l in cLayers:
            layer = l.instantiate()
            name = l.get_name()
            # ensure unique name
            if name in layers:
                basename = name
                idx = 1
                while name in layers:
                    name = basename + "_%d"%idx
                    idx += 1
            l.name = name
            layers[l.name] = layer
        return layers, cLayers

    def build(self):
        layers, cLayers = self.get_named_layers()

        weight_manager = BufferManager()
        for name, l in layers.items()[1:-1]:
            sources = {}
            sinks = {name: (l.get_param_size, l.create_param_view)}
            weight_manager.add(sources, sinks)

        intern_manager = BufferManager()
        for name, l in layers.items()[1:-1]:
            sources = {}
            sinks = {name: (l.get_internal_state_size, l.create_internal_view)}
            intern_manager.add(sources, sinks)

        output_manager = BufferManager()
        for layer in cLayers[:-1]:
            lset, rset = self.get_forward_closure(layer)
            assert len(lset)==len(rset)==1, "Complicated Architectures not supported yet"
            sources = dict()
            for n in rset:
                l = layers[n.name]
                sources[n.name] = (l.get_input_size, l.create_input_view)
            sinks = dict()
            for n in lset:
                l = layers[n.name]
                sinks[n.name] = (l.get_output_size, l.create_output_view)

            output_manager.add(sources, sinks)

        net = Network(layers, weight_manager, intern_manager, output_manager)
        return net

