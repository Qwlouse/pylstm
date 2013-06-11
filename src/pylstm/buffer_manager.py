#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
import wrapper


class BufferHub(object):
    def __init__(self, sources, sinks, slices=1, batches=1, con_table=None):
        # only full connection supported so far
        assert con_table is None or np.all(con_table == 1), \
            "Only full connections supported so far."
        self.sources = sources
        self.sinks = sinks
        self.buffer = None
        self.views = None
        assert slices >= 1
        assert batches >= 1
        self.slice_count = slices
        self.batch_count = batches

    def set_dimensions(self, slice_count, batch_count):
        assert slice_count > 0
        assert batch_count > 0
        self.slice_count = slice_count
        self.batch_count = batch_count
        self.views = None

    def get_size(self):
        # with full connections the size is determined by the sum of all sources
        # or by the size of any single sink
        if len(self.sinks) > 0:
            # get a sink
            sg, vf = self.sinks.values()[0]
            return sg(self.slice_count, self.batch_count)
        else:
            return sum(sg(self.slice_count, self.batch_count)
                       for (sg, vf) in self.sources.values())

    def set_buffer(self, buffer_view):
        self.buffer = buffer_view
        self.buffer = buffer_view.reshape(self.slice_count, self.batch_count, -1)
        self.views = None

    def _lay_out_source_buffers(self):
        start = 0
        for n, (sg, vf) in self.sources.items():
            assert sg(self.slice_count, self.batch_count) % (self.slice_count * self.batch_count) == 0, "buffer: %s with %d %d needs %d"%(self.buffer.shape(), self.slice_count, self.batch_count,  sg(self.slice_count, self.batch_count))
            size = sg(self.slice_count, self.batch_count) / self.slice_count / self.batch_count
            self.views[n] = vf(self.buffer.feature_slice(start, start + size),
                               self.slice_count, self.batch_count)
            start += size
        return start * self.slice_count * self.batch_count

    def create_views(self):
        self.views = {}
        source_size = self._lay_out_source_buffers()
        assert source_size == 0 or (source_size == self.get_size())

        for n, (sg, vf) in self.sinks.items():
            size = sg(self.slice_count, self.batch_count)
            assert size == source_size
            self.views[n] = vf(self.buffer, self.slice_count, self.batch_count)

    def get_buffer(self, name):
        assert self.buffer is not None
        if self.views is None:
            self.create_views()
        return self.views[name]


class BufferManager(object):
    def __init__(self, slice_count=1, batch_count=1):
        assert slice_count > 0
        assert batch_count > 0
        self.slice_count = slice_count
        self.batch_count = batch_count
        self.buffer = None
        self.views_ready = False

        self.buffers_by_source = {}
        self.buffers_by_sinks = {}
        self.buffer_hubs = []

    def add(self, sources, sinks, con_table=None):
        self.buffer = None
        bh = BufferHub(sources, sinks,
                       self.slice_count, self.batch_count,
                       con_table)
        self.buffer_hubs.append(bh)
        for n in sources:
            self.buffers_by_source[n] = bh
        for n in sinks:
            self.buffers_by_sinks[n] = bh

    def set_dimensions(self, slice_count, batch_count, force_resize=False):
        assert slice_count > 0
        assert batch_count > 0
        # todo: here is some room for optimization:
        # if the dims didn't change and force_resize is false then we don't
        # need to do all of the following steps
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
