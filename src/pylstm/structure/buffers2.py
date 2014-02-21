import numpy as np
import itertools


def get_forward_closure(layer, architecture):
    """
    For a given layer return two sets of layer names such that:
      * the given layer is in the source_set
      * the sink_set contains all the target layers of the source_set
      * the source_set contains all the source layers of the sink_set

    Parameters
    ----------
    layer : string
        The name of the layer to start the forward closure from.
    architecture : dict
        Extended architecture of the network mapping the layer name to the layer
        description. The description has to be a dictionary containing lists
        for 'sources' and 'targets'.
    Returns
    -------
    source_set : set
        A set of layer names containing the initial layer and all sources of all
        layers in the sink_set.
    sink_set : set
        A set of layer names containing all the targets for all the layers from
        the source_set.

    See Also
    --------
    extend_architecture_info : Returns an extended version of an architecture
    including 'sources' for all layers.
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


def permute_sources(source_list, connection_table):
    """
    Given a list of sources and a connection table, find a permutation of the
    sources, such that they can be connected to the sinks via a single buffer.
    """
    # systematically try all permutations until one satisfies the condition
    final_permutation = None
    for perm in itertools.permutations(range(len(source_list))):
        ct = connection_table[perm]
        if can_be_connected_with_single_buffer(ct):
            final_permutation = perm
            break
    if final_permutation is None:
        raise RuntimeError("Impossible")

    # apply the permutation
    source_list = [source_list[i] for i in final_permutation]
    connection_table = connection_table[final_permutation]
    return source_list, connection_table


def can_be_connected_with_single_buffer(connection_table):
    """
    Check for a connection table if it represents a layout that can be realized
    by a single buffer. This is equivalent to checking if in every column of the
    table all the ones form a connected block.
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