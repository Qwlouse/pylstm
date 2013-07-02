#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from buffer_manager import BufferManager
from layers import DummyLayer, InvalidArchitectureError
from network import Network
from pylstm.error_functions import MeanSquaredError
import numpy as np


class NetworkBuilder(object):
    def __init__(self):
        self.input_layer = None
        self.output = DummyLayer(0, "Output")
        self.error_func = MeanSquaredError()

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
        For a given layer return two sorted lists of layer names such that:
          * the given layer is in the source_set
          * the sink_set contains all the target layers of the source_set
          * the source_set contains all the source layers of the sink_set
        """
        # grow the two sets
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
        # turn into sorted lists
        source_list = sorted([l for l in source_set], key=lambda x: x.name)
        sink_list = sorted([l for l in sink_set], key=lambda x: x.name)
        # set up connection table
        connection_table = np.zeros((len(source_list), len(sink_list)))
        for i, source in enumerate(source_list):
            for sink in source.targets:
                connection_table[i, sink_list.index(sink)] = 1
        # convert to lists of names
        source_name_list = [s.name for s in source_list]
        sink_name_list = [s.name for s in sink_list]
        return source_name_list, sink_name_list, connection_table

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

        param_manager = BufferManager()
        fwd_state_manager = BufferManager()
        bwd_state_manager = BufferManager()
        for name, l in layers.items()[1:-1]:
            sources = {name: (l.get_parameter_size, l.create_param_view)}
            param_manager.add(sources, {})

            sources = {name: (l.get_fwd_state_size,
                              l.create_fwd_state)}
            fwd_state_manager.add(sources, {})

            sources = {name: (l.get_bwd_state_size,
                              l.create_bwd_state)}
            bwd_state_manager.add(sources, {})

        in_out_manager = BufferManager()
        delta_manager = BufferManager()
        for layer in cLayers[:-1]:
            source_list, sink_list, con_table = self.get_forward_closure(layer)
            assert np.all(con_table == 1), \
                "Sparse Architectures not supported yet"
            sinks = {n: (layers[n].get_input_buffer_size,
                         layers[n].create_input_view) for n in sink_list}
            sources = {n: (layers[n].get_output_buffer_size,
                           layers[n].create_output_view) for n in source_list}

            in_out_manager.add(sources, sinks, con_table)
            delta_manager.add(sources, sinks, con_table)

        param_manager.set_dimensions(1, 1)

        net = Network(layers, param_manager, fwd_state_manager, in_out_manager,
                      bwd_state_manager, delta_manager, self.error_func)
        return net