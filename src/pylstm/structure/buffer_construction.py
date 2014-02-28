from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
import itertools
from buffer_manager import BufferManager


def create_param_manager(layers):
    """
    @type layers: dict[unicode, pylstm.wrapper.py_layers.BaseLayer]
    @rtype: buffer_manager.BufferManager
    """
    param_manager = BufferManager()
    for name, l in layers.items()[1:]:
        sources = {name: (l.get_parameter_size, l.create_param_view)}
        param_manager.add(sources, {})
    return param_manager


def create_fwd_state_manager(layers):
    """
    @type layers: dict[unicode, pylstm.wrapper.py_layers.BaseLayer]
    @rtype: buffer_manager.BufferManager
    """
    fwd_state_manager = BufferManager()
    for name, l in layers.items()[1:]:
        sources = {name: (l.get_fwd_state_size,
                          l.create_fwd_state)}
        fwd_state_manager.add(sources, {})

    return fwd_state_manager


def create_bwd_state_manager(layers):
    """
    @type layers: dict[unicode, pylstm.wrapper.py_layers.BaseLayer]
    @rtype: buffer_manager.BufferManager
    """
    bwd_state_manager = BufferManager()
    for name, l in layers.items()[1:]:
        sources = {name: (l.get_bwd_state_size,
                          l.create_bwd_state)}
        bwd_state_manager.add(sources, {})
    return bwd_state_manager


def create_in_out_manager(extended_architecture, layers):
    """
    @type extended_architecture: dict
    @type layers: dict[unicode, pylstm.wrapper.py_layers.BaseLayer]
    @rtype: buffer_manager.BufferManager
    """
    in_out_manager = BufferManager()

    for layer in extended_architecture.keys():
        source_set, sink_set = get_forward_closure(layer, extended_architecture)
        source_list, sink_list, connection_table = set_up_connection_table(
            source_set, sink_set, extended_architecture)
        perm = permute_rows(connection_table)
        source_list = [source_list[i] for i in perm]
        connection_table = np.atleast_2d(connection_table[perm])

        source_getters = OrderedDict()
        for n in source_list:
            source_getters[n] = (layers[n].get_output_buffer_size,
                                 layers[n].create_output_view)

        sink_getters = OrderedDict()
        for n in sink_list:
            sink_getters[n] = (layers[n].get_input_buffer_size,
                               layers[n].create_input_view)

        in_out_manager.add(source_getters, sink_getters, connection_table)
    return in_out_manager


def get_forward_closure(layer, architecture):
    """
    For a given layer return two sets of layer names such that:
      - the given layer is in the source_set
      - the sink_set contains all the target layers of the source_set
      - the source_set contains all the source layers of the sink_set

    @type layer: unicode
    @param layer: The name of the layer to start the forward closure from.
    @type architecture: dict
    @param architecture: Extended architecture of the network mapping the
        layer name to the layer description. The description has to be a
        dictionary containing lists for 'sources' and 'targets'.

    @rtype: (set, set)
    @return: A tuple (source_set, sink_set) where source_set is set of
        layer names containing the initial layer and all sources of all layers
        in the sink_set. And sink_set is a set of layer names containing all the
        targets for all the layers from the source_set.
    """
    source_set = {layer}
    sink_set = set(architecture[layer]['targets'])
    growing = True
    while growing:
        growing = False
        new_source_set = {s for l in sink_set
                          for s in architecture[l]['sources']}
        new_sink_set = {t for l in source_set
                        for t in architecture[l]['targets']}
        if len(new_source_set) > len(source_set) or\
                len(new_sink_set) > len(sink_set):
            growing = True
            source_set = new_source_set
            sink_set = new_sink_set
    return source_set, sink_set


def set_up_connection_table(sources, sinks, architecture):
    """
    Given a forward closure and the architecture constructs the
    connection table.
    @type sources: set[unicode]
    @type sinks: set[unicode]
    @type architecture: dict

    @rtype: (list, list, np.ndarray)
    """
    # turn into sorted lists
    source_list = sorted([l for l in sources])
    sink_list = sorted([l for l in sinks])
    # set up connection table
    connection_table = np.zeros((len(source_list), len(sink_list)))
    for i, source in enumerate(source_list):
        for sink in architecture[source]['targets']:
            connection_table[i, sink_list.index(sink)] = 1

    return source_list, sink_list, connection_table


def permute_rows(connection_table):
    """
    Given a list of sources and a connection table, find a permutation of the
    sources, such that they can be connected to the sinks via a single buffer.
    @type connection_table: np.ndarray
    @rtype: list[int]
    """
    # systematically try all permutations until one satisfies the condition
    final_permutation = None
    for perm in itertools.permutations(range(connection_table.shape[0])):
        ct = np.atleast_2d(connection_table[perm])
        if can_be_connected_with_single_buffer(ct):
            final_permutation = perm
            break
    if final_permutation is None:
        raise RuntimeError("Impossible")

    return final_permutation


def can_be_connected_with_single_buffer(connection_table):
    """
    Check for a connection table if it represents a layout that can be realized
    by a single buffer. This is equivalent to checking if in every column of the
    table all the ones form a connected block.
    @type connection_table: np.ndarray
    @rtype: bool
    """
    for i in range(connection_table.shape[1]):
        region_started = False
        region_stopped = False
        for j in range(connection_table.shape[0]):
            if not region_started and connection_table[j, i]:
                region_started = True
            elif region_started and not region_stopped and not connection_table[j, i]:
                region_stopped = True
            elif region_stopped and connection_table[j, i]:
                return False
    return True