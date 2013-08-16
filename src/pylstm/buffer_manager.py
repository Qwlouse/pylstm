#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
import wrapper


class BufferHub(object):
    def __init__(self, sources, sinks, con_table=None):
        # only full connection supported so far
        assert con_table is None or np.all(con_table == 1), \
            "Only full connections supported so far."
        self.sources = sources
        self.sinks = sinks
        self.buffer = None
        self.views = None
        self.slice_count = None
        self.batch_count = None
        # evaluate size lazily
        self.size = None

    def set_dimensions(self, slice_count, batch_count):
        if slice_count == self.slice_count and batch_count == self.batch_count:
            return
        assert slice_count > 0
        assert batch_count > 0
        self.slice_count = slice_count
        self.batch_count = batch_count
        self.views = None
        self.size = None

    def get_size(self):
        # with full connections the size is determined by the sum of all sources
        # or by the size of any single sink
        assert self.slice_count is not None
        assert self.batch_count is not None
        if self.size is None:
            if len(self.sinks) > 0:
                # get a sink
                sg, vf = self.sinks.values()[0]
                self.size = sg(self.slice_count, self.batch_count)
            else:
                self.size = sum(sg(self.slice_count, self.batch_count)
                                for (sg, vf) in self.sources.values())
        return self.size

    def set_buffer(self, buffer_view):
        assert self.slice_count is not None
        assert self.batch_count is not None
        self.buffer = buffer_view
        self.buffer = buffer_view.reshape(self.slice_count, self.batch_count, -1)
        self.views = None

    def _lay_out_source_buffers(self):
        assert self.slice_count is not None
        assert self.batch_count is not None
        assert self.buffer is not None
        start = 0
        for n, (sg, vf) in self.sources.items():
            s = sg(self.slice_count, self.batch_count)
            assert s % (self.slice_count * self.batch_count) == 0, "buffer: %s with %d %d needs %d"%(self.buffer.shape(), self.slice_count, self.batch_count,  sg(self.slice_count, self.batch_count))
            size = s / self.slice_count / self.batch_count
            self.views[n] = vf(self.buffer.feature_slice(start, start + size),
                               self.slice_count, self.batch_count)
            start += size
        return start * self.slice_count * self.batch_count

    def create_views(self):
        assert self.slice_count is not None
        assert self.batch_count is not None
        assert self.buffer is not None
        self.views = {}
        source_size = self._lay_out_source_buffers()
        assert source_size == 0 or (source_size == self.get_size())

        for n, (sg, vf) in self.sinks.items():
            size = sg(self.slice_count, self.batch_count)
            assert size == source_size
            self.views[n] = vf(self.buffer, self.slice_count, self.batch_count)

    def get_buffer(self, name):
        assert self.buffer is not None
        assert self.slice_count is not None
        assert self.batch_count is not None
        if self.views is None:
            self.create_views()
        return self.views[name]


class BufferManager(object):
    def __init__(self):
        self.slice_count = None
        self.batch_count = None
        self.buffer = None
        self.views_ready = False

        self.buffers_by_source = {}
        self.buffers_by_sinks = {}
        self.buffer_hubs = []

    def add(self, sources, sinks, con_table=None):
        self.buffer = None
        bh = BufferHub(sources, sinks, con_table)
        self.buffer_hubs.append(bh)
        for n in sources:
            self.buffers_by_source[n] = bh
        for n in sinks:
            self.buffers_by_sinks[n] = bh

    def set_dimensions(self, slice_count, batch_count, force_resize=False):
        if (slice_count == self.slice_count and
                batch_count == self.batch_count and
                not force_resize):
            return
        assert slice_count > 0
        assert batch_count > 0
        self.slice_count = slice_count
        self.batch_count = batch_count
        for bh in self.buffer_hubs:
            bh.set_dimensions(slice_count, batch_count)
        new_size = self.calculate_size()
        if self.buffer and (new_size > len(self.buffer) or force_resize):
            self.buffer = None
        self.views_ready = False

    def calculate_size(self):
        return sum(bh.get_size() for bh in self.buffer_hubs)

    def initialize_buffer(self, buffer_view=None):
        total_size = self.calculate_size()
        if buffer_view is None:
            self.buffer = wrapper.Matrix(total_size)
        else:
            assert len(buffer_view) >= total_size
            self.buffer = buffer_view.reshape(-1, 1, 1)
        self.views_ready = False

    def lay_out_buffer_hubs(self):
        assert self.buffer is not None
        param_start = 0
        for bh in self.buffer_hubs:
            param_size = bh.get_size()
            bview = self.buffer[param_start: param_start + param_size]
            bh.set_buffer(bview)
            param_start += param_size
        self.views_ready = True

    def ensure_initialization(self):
        if self.buffer is None:
            self.initialize_buffer()
        if not self.views_ready:
            self.lay_out_buffer_hubs()

    def get_source_view(self, name):
        self.ensure_initialization()
        return self.buffers_by_source[name].get_buffer(name)

    def get_sink_view(self, name):
        self.ensure_initialization()
        return self.buffers_by_sinks[name].get_buffer(name)

    def clear_buffer(self):
        self.ensure_initialization()
        self.buffer.set_all_elements_to(0.0)


def get_forward_closure(layer, extended_architecture):
    """
    For a given layer return two sorted lists of layer names such that:
      * the given layer is in the source_set
      * the sink_set contains all the target layers of the source_set
      * the source_set contains all the source layers of the sink_set
    """
    # grow the two sets
    source_set = {layer}
    sink_set = set(extended_architecture[layer]['targets'])
    growing = True
    while growing:
        growing = False
        new_source_set = {s for l in sink_set
                          for s in extended_architecture[l]['sources']}
        new_sink_set = {t for l in source_set
                        for t in extended_architecture[l]['targets']}
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
        for sink in extended_architecture[source]['targets']:
            connection_table[i, sink_list.index(sink)] = 1
    # convert to lists of names
    return source_list, sink_list, connection_table


def create_param_manager(layers):
    param_manager = BufferManager()
    fwd_state_manager = BufferManager()
    bwd_state_manager = BufferManager()
    for name, l in layers.items()[1:]:
        sources = {name: (l.get_parameter_size, l.create_param_view)}
        param_manager.add(sources, {})

        sources = {name: (l.get_fwd_state_size,
                          l.create_fwd_state)}
        fwd_state_manager.add(sources, {})

        sources = {name: (l.get_bwd_state_size,
                          l.create_bwd_state)}
        bwd_state_manager.add(sources, {})
    return param_manager


def create_in_out_manager(extended_architecture, layers):
    in_out_manager = BufferManager()

    for layer in extended_architecture.keys():
        source_list, sink_list, con_table = get_forward_closure(layer, extended_architecture)
        assert np.all(con_table == 1), "Sparse Architectures not supported yet"
        sinks = {n: (layers[n].get_input_buffer_size,
                     layers[n].create_input_view) for n in sink_list}
        sources = {n: (layers[n].get_output_buffer_size,
                       layers[n].create_output_view) for n in source_list}

        in_out_manager.add(sources, sinks, con_table)
    return in_out_manager


def create_fwd_state_manager(layers):
    fwd_state_manager = BufferManager()
    for name, l in layers.items()[1:]:
        sources = {name: (l.get_fwd_state_size,
                          l.create_fwd_state)}
        fwd_state_manager.add(sources, {})

    return fwd_state_manager


def create_bwd_state_manager(layers):
    bwd_state_manager = BufferManager()
    for name, l in layers.items()[1:]:
        sources = {name: (l.get_bwd_state_size,
                          l.create_bwd_state)}
        bwd_state_manager.add(sources, {})
    return bwd_state_manager
