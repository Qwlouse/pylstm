#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from copy import deepcopy
from buffer_manager import BufferManager
from layers import Input, Output, InvalidArchitectureError
from network import Network
from pylstm.error_functions import MeanSquaredError
import numpy as np
from pylstm.layers.construction_layer import instantiate_layer


class NetworkBuilder(object):
    def __init__(self):
        self.input_layer = None
        self.output = Output(0, "Output")
        self.error_func = MeanSquaredError

    def input(self, size=None):
        if size:
            self.input_layer = Input(size, "Input")
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
        for layer in cLayers[:-1]:
            source_list, sink_list, con_table = self.get_forward_closure(layer)
            assert np.all(con_table == 1), \
                "Sparse Architectures not supported yet"
            sinks = {n: (layers[n].get_input_buffer_size,
                         layers[n].create_input_view) for n in sink_list}
            sources = {n: (layers[n].get_output_buffer_size,
                           layers[n].create_output_view) for n in source_list}

            in_out_manager.add(sources, sinks, con_table)

        param_manager.set_dimensions(1, 1)

        architecture = OrderedDict()

        for l in cLayers[:-1]:
            layer_entry = {
                'size': l.out_size,
                'type': l.layer_type,
                'targets': [t.get_name() for t in l.targets],
            }
            if l.layer_kwargs:
                layer_entry['kwargs'] = l.layer_kwargs
            architecture[l.get_name()] = layer_entry

        net = Network(layers, param_manager, fwd_state_manager, in_out_manager,
                      bwd_state_manager, self.error_func,
                      architecture=architecture)

        return net


def find_input_layer(some_layer):
    all_layers = some_layer.collect_all_connected_layers()
    # find input layer(s)
    input_layers = [l for l in all_layers if l.layer_type == 'Input']
    assert len(input_layers) == 1, \
        "Found %d Inputs, but has to be 1." % len(input_layers)

    input_layer = input_layers[0]
    assert len(input_layer.sources) == 0, \
        "Input is not allowed to have sources."

    source_layers = [l for l in all_layers if len(l.sources) == 0]
    assert len(source_layers) == 1 and source_layers[0] == input_layer,\
        "Only the Input is allowed to have empty sources list!"

    return input_layers[0]


def get_sorted_layers(input_layer):
    # gather all the layers
    layers = set(input_layer.traverse_targets_tree())
    # sort them by depth
    input_layer.depth = 0
    return sorted(layers, key=lambda l: l.get_depth())


def ensure_unique_output_layer(layers):
    output_layers = [l for l in layers if l.layer_type == 'Output']
    assert len(output_layers) == 1, \
        "Found %d Outputs, but has to be 1." % len(output_layers)

    output_layer = output_layers[0]
    assert len(output_layer.targets) == 0, \
        "Output is not allowed to have targets."

    sink_layers = [l for l in layers if len(l.targets) == 0]
    assert len(sink_layers) == 1 and sink_layers[0] == output_layer,\
        "Only the Output is allowed to have empty targets list!"


def ensure_unique_names_for_layers(layers):
    layer_map = {}
    for l in layers:
        name = l.get_name()
        # ensure unique name
        if name in layer_map or (name + '_1') in layer_map:
            basename = name
            idx = 2
            while name in layers:
                name = basename + "_%d" % idx
                idx += 1

            # rename original conflicting layer
            conflicting_layer = layer_map[basename]
            conflicting_layer.name = basename + '_1'
            del layer_map[basename]
            layer_map[conflicting_layer.name] = conflicting_layer

        l.name = name
        layer_map[name] = l
    return layer_map


def build_architecture_from_layers_list(layers):
    architecture = OrderedDict()
    for l in layers[:-1]:
        layer_entry = {
            'size': l.out_size,
            'type': l.layer_type,
            'targets': [t.get_name() for t in l.targets],
        }
        if l.layer_kwargs:
            layer_entry['kwargs'] = l.layer_kwargs
        architecture[l.get_name()] = layer_entry
    return architecture


def create_architecture_from_layers(some_layer):
    input_layer = find_input_layer(some_layer)
    layers = get_sorted_layers(input_layer)
    ensure_unique_output_layer(layers)
    ensure_unique_names_for_layers(layers)
    architecture = build_architecture_from_layers_list(layers)
    return architecture


def extend_architecture_info(architecture):
    extended_architecture = deepcopy(architecture)  # do not modify original

    extended_architecture['Output'] = {
        'size': 0,
        'type': 'Output',
        'targets': []
    }

    # add in_size to every layer:
    for n, l in extended_architecture.items():
        l['in_size'] = 0
        l['sources'] = []
        if 'kwargs' not in l:
            l['kwargs'] = {}

    # calculate correct values
    for n, l in extended_architecture.items():
        for target_name in l['targets']:
            target = extended_architecture[target_name]
            target['sources'].append(n)
            target['in_size'] += l['size']
    return extended_architecture


def instantiate_layers_from_architecture(architecture):
    # instantiate layers
    layers = OrderedDict()
    for name, prop in architecture.items():
        layer = instantiate_layer(prop['type'], prop['in_size'], prop['size'],
                                  prop['kwargs'])
        layers[name] = layer

    return layers


def create_param_manager(layers):
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
    return param_manager


def create_fwd_state_manager(layers):
    fwd_state_manager = BufferManager()
    for name, l in layers.items()[1:-1]:
        sources = {name: (l.get_fwd_state_size,
                          l.create_fwd_state)}
        fwd_state_manager.add(sources, {})

    return fwd_state_manager


def create_bwd_state_manager(layers):
    bwd_state_manager = BufferManager()
    for name, l in layers.items()[1:-1]:
        sources = {name: (l.get_bwd_state_size,
                          l.create_bwd_state)}
        bwd_state_manager.add(sources, {})
    return bwd_state_manager


def get_forward_closure(layer, arch):
    """
    For a given layer return two sorted lists of layer names such that:
      * the given layer is in the source_set
      * the sink_set contains all the target layers of the source_set
      * the source_set contains all the source layers of the sink_set
    """
    # grow the two sets
    source_set = {layer}
    sink_set = set(arch[layer]['targets'])
    growing = True
    while growing:
        growing = False
        new_source_set = {s for l in sink_set for s in arch[l]['sources']}
        new_sink_set = {t for l in source_set for t in arch[l]['targets']}
        if len(new_source_set) > len(source_set) or\
                len(new_sink_set) > len(sink_set):
            growing = True
            source_set = new_source_set
            sink_set = new_sink_set
    # turn into sorted lists
    source_list = sorted([l for l in source_set])
    sink_list = sorted([l for l in sink_set])
    # set up connection table
    connection_table = np.zeros((len(source_list), len(sink_list)))
    for i, source in enumerate(source_list):
        for sink in arch[source]['targets']:
            connection_table[i, sink_list.index(sink)] = 1
    # convert to lists of names
    return source_list, sink_list, connection_table


def create_in_out_manager(ex_arch, layers):
    in_out_manager = BufferManager()

    for layer in ex_arch.keys()[:-1]:
        source_list, sink_list, con_table = get_forward_closure(layer, ex_arch)
        assert np.all(con_table == 1), "Sparse Architectures not supported yet"
        sinks = {n: (layers[n].get_input_buffer_size,
                     layers[n].create_input_view) for n in sink_list}
        sources = {n: (layers[n].get_output_buffer_size,
                       layers[n].create_output_view) for n in source_list}

        in_out_manager.add(sources, sinks, con_table)
    return in_out_manager


def build_network_from_architecture(architecture):
    #TODO: validate architecture
    ex_arch = extend_architecture_info(architecture)
    layers = instantiate_layers_from_architecture(ex_arch)
    param_manager = create_param_manager(layers)
    param_manager.set_dimensions(1, 1)
    fwd_state_manager = create_fwd_state_manager(layers)
    bwd_state_manager = create_bwd_state_manager(layers)
    in_out_manager = create_in_out_manager(ex_arch, layers)

    net = Network(layers,
                  param_manager,
                  fwd_state_manager,
                  in_out_manager,
                  bwd_state_manager,
                  error_func=MeanSquaredError,
                  architecture=architecture)
    return net