#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from buffer_manager import BufferManager
from layers import DummyLayer, InvalidArchitectureError
from network import Network
import wrapper


class NetworkBuilder(object):
    def __init__(self):
        self.input_layer = None
        self.output = DummyLayer(0, "Output")

    def input(self, size=None):
        if size:
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
        for a given layer return two sets of layer names such that:
          * the given layer is in the source_set
          * the sink_set contains all the target layers of the source_set
          * the source_set contains all the source layers of the sink_set
        """
        source_set = {layer}
        sink_set = set(layer.targets)
        growing = True
        while growing:
            growing = False
            new_source_set = {s for l in sink_set for s in l.sources}
            new_sink_set = {t for l in source_set for t in l.targets}
            if len(new_source_set) > len(source_set) or\
                    len(new_sink_set) > len(sink_set):
                growing = True
                source_set = new_source_set
                sink_set = new_sink_set
        return {l.name for l in source_set}, {l.name for l in sink_set}

    def create_buffer(self, size):
        return wrapper.BufferView(1, 1, size)

    def get_named_layers(self):
        # instantiate all the layers with names
        cLayers = self.get_sorted_layers()
        assert cLayers[0] is self.input_layer
        assert cLayers[-1] is self.output
        self.output.out_size = self.output.get_input_size()
        layers = OrderedDict()
        for l in cLayers:
            layer = l.instantiate()
            name = l.get_name()
            # ensure unique name
            if name in layers:
                basename = name
                idx = 1
                while name in layers:
                    name = basename + "_%d" % idx
                    idx += 1
            l.name = name
            layers[l.name] = layer
        return layers, cLayers

    def build(self):
        layers, cLayers = self.get_named_layers()

        weight_manager = BufferManager()
        intern_manager = BufferManager()
        intern_delta_manager = BufferManager()
        for name, l in layers.items()[1:-1]:
            sources = {name: (l.get_param_size, l.create_param_view)}
            weight_manager.add(sources, {})

            sources = {name: (l.get_internal_state_size,
                              l.create_internal_view)}
            intern_manager.add(sources, {})

            sources = {name: (l.get_internal_error_state_size,
                              l.create_internal_error_view)}
            intern_delta_manager.add(sources, {})

        in_out_manager = BufferManager()
        delta_manager = BufferManager()
        for layer in cLayers[:-1]:
            source_set, sink_set = self.get_forward_closure(layer)
            assert len(source_set) == len(sink_set) == 1, \
                "Complicated Architectures not supported yet"
            sinks = {n: (layers[n].get_input_buffer_size,
                         layers[n].create_input_view) for n in sink_set}
            sources = {n: (layers[n].get_output_buffer_size,
                           layers[n].create_output_view) for n in source_set}

            in_out_manager.add(sources, sinks)
            delta_manager.add(sources, sinks)

        net = Network(layers, weight_manager, intern_manager, in_out_manager,
                      intern_delta_manager, delta_manager)
        return net