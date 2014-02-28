#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
from .. import wrapper


class BufferHub(object):
    def __init__(self, sources, sinks, con_table=None):
        self.sources = sources
        self.sinks = sinks
        self.con_table = con_table
        self.full_sink = None
        if con_table and sinks:
            # find a sink that connects to all sources
            # this will later speed up the size computation
            nr_sources = con_table.shape[0]  # == len(sources)
            full_sinks = np.flatnonzero(np.sum(con_table, axis=0) == nr_sources)
            if len(full_sinks):
                self.full_sink = self.sinks.values[full_sinks[0]]

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
        # The size is determined by the sum of all sources
        # or by the size of a single full sink
        assert self.slice_count is not None
        assert self.batch_count is not None
        if self.size is None:
            if self.full_sink is not None:
                sg, vf = self.full_sink
                self.size = sg(self.slice_count, self.batch_count)
            else:
                self.size = sum(sg(self.slice_count, self.batch_count)
                                for (sg, vf) in self.sources.values())
        return self.size

    def set_buffer(self, buffer_view):
        assert self.slice_count is not None
        assert self.batch_count is not None
        self.buffer = buffer_view
        self.buffer = buffer_view.reshape(self.slice_count,
                                          self.batch_count, -1)
        self.views = None

    def _lay_out_source_buffers(self):
        assert self.slice_count is not None
        assert self.batch_count is not None
        assert self.buffer is not None
        start = 0
        intervals = [0]
        for n, (sg, vf) in self.sources.items():
            s = sg(self.slice_count, self.batch_count)
            assert s % (self.slice_count * self.batch_count) == 0, \
                "buffer: %s with %d %d needs %d" % (self.buffer.shape(),
                                                    self.slice_count,
                                                    self.batch_count,
                                                    sg(self.slice_count,
                                                       self.batch_count))
            size = s / self.slice_count / self.batch_count
            self.views[n] = vf(self.buffer.feature_slice(start, start + size),
                               self.slice_count, self.batch_count)
            start += size
            intervals.append(start)
        source_size = start * self.slice_count * self.batch_count
        assert source_size == 0 or (source_size == self.get_size())
        return intervals

    def _lay_out_sink_buffers(self, intervals):
        for i, (n, (sg, vf)) in enumerate(self.sinks.items()):
            connectivity = self.con_table[:, i]
            start_idx = np.argmax(connectivity)
            stop_idx = len(connectivity) - int(np.argmax(connectivity[::-1]))
            start, stop = intervals[start_idx], intervals[stop_idx]
            self.views[n] = vf(self.buffer.feature_slice(start, stop),
                               self.slice_count, self.batch_count)
            size = sg(self.slice_count, self.batch_count)
            assert size == (stop - start) * self.slice_count * self.batch_count

    def create_views(self):
        assert self.slice_count is not None
        assert self.batch_count is not None
        assert self.buffer is not None
        self.views = {}
        intervals = self._lay_out_source_buffers()
        self._lay_out_sink_buffers(intervals)

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

        self.buffer_hubs_by_source = {}
        self.buffer_hubs_by_sink = {}
        self.buffer_hubs = []

    def add(self, sources, sinks, con_table=None):
        self.buffer = None
        bh = BufferHub(sources, sinks, con_table)
        self.buffer_hubs.append(bh)
        for n in sources:
            self.buffer_hubs_by_source[n] = bh
        for n in sinks:
            self.buffer_hubs_by_sink[n] = bh

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
        return self.buffer_hubs_by_source[name].get_buffer(name)

    def get_sink_view(self, name):
        self.ensure_initialization()
        return self.buffer_hubs_by_sink[name].get_buffer(name)

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
