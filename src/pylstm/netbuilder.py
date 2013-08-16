#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from copy import deepcopy
from buffer_manager import create_param_manager, create_fwd_state_manager
from buffer_manager import create_bwd_state_manager, create_in_out_manager
from network import Network
from pylstm.error_functions import MeanSquaredError
from pylstm.layers.construction_layer import instantiate_layer


def find_input_layer(some_layer):
    all_layers = some_layer.collect_all_connected_layers()
    # find input layer(s)
    input_layers = [l for l in all_layers if l.layer_type == 'InputLayer']
    assert len(input_layers) == 1, \
        "Found %d InputLayers, but has to be 1." % len(input_layers)

    input_layer = input_layers[0]
    assert len(input_layer.sources) == 0, \
        "InputLayer is not allowed to have sources."

    source_layers = [l for l in all_layers if len(l.sources) == 0]
    assert len(source_layers) == 1 and source_layers[0] == input_layer,\
        "Only the InputLayer is allowed to have empty sources list!"

    return input_layers[0]


def get_topologically_sorted_layers(input_layer):
    layers = set(input_layer.traverse_targets_tree())
    input_layer.depth = 0
    return sorted(layers, key=lambda l: l.get_depth())


def ensure_unique_output_layer(layers):
    output_layers = [l for l in layers if len(l.targets) == 0]
    assert len(output_layers) == 1,\
        "Only one Layer is allowed to have empty targets list! But found %s." %\
        str(output_layers)


def ensure_unique_names_for_layers(layers):
    layer_map = {}
    for l in layers:
        name = l.get_name()
        if name in layer_map or (name + '_1') in layer_map:
            basename = name
            idx = 2
            while name in layer_map:
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
    for l in layers:
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
    layers = get_topologically_sorted_layers(input_layer)
    ensure_unique_output_layer(layers)
    ensure_unique_names_for_layers(layers)
    architecture = build_architecture_from_layers_list(layers)
    return architecture


def validate_architecture(architecture):
    # schema
    for name, info in architecture.items():
        assert 'size' in info and isinstance(info['size'], int)
        assert 'type' in info and isinstance(info['type'], basestring)
        assert 'targets' in info and isinstance(info['targets'], list)

    # has InputLayer
    assert 'InputLayer' in architecture
    assert architecture['InputLayer']['type'] == 'InputLayer'

    # has only one InputLayer
    inputs_by_type = [l for l in architecture.values()
                      if l['type'] == 'InputLayer']
    assert len(inputs_by_type) == 1

    # no sources for InputLayer
    input_sources = [l for l in architecture.values()
                     if 'InputLayer' in l['targets']]
    assert len(input_sources) == 0

    # only 1 output
    outputs = [l for l in architecture.values() if not l['targets']]
    assert len(outputs) == 1

    # no loops and topologically sorted
    seen_names = set()
    for name, info in reversed(architecture.items()):
        for t in info['targets']:
            assert t in seen_names
        seen_names.add(name)


def extend_architecture_info(architecture):
    extended_architecture = deepcopy(architecture)  # do not modify original

    for n, l in extended_architecture.items():
        l['in_size'] = 0
        l['sources'] = []
        if 'kwargs' not in l:
            l['kwargs'] = {}

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


def build_network_from_architecture(architecture):
    validate_architecture(architecture)
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


def build_net(some_layer):
    arch = create_architecture_from_layers(some_layer)
    net = build_network_from_architecture(arch)
    return net