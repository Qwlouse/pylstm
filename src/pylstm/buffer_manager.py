#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import pylstm_wrapper

class BufferManager(object):
    def __init__(self, slice_count=1, batch_count=1):
        assert slice_count > 0
        assert batch_count > 0
        self.slice_count = slice_count
        self.batch_count = batch_count
        self.size_getters = {}
        self.view_factories = {}
        self.buffer = None
        self.views = None

    def add(self, name, size_getter, view_factories):
        self.buffer = None
        self.size_getters[name] = size_getter
        self.view_factories[name] = view_factories

    def set_dimensions(self, slice_count, batch_count, force_resize=False):
        assert slice_count > 0
        assert batch_count > 0
        self.slice_count = slice_count
        self.batch_count = batch_count
        new_size = self.calculate_size()
        if self.buffer and (new_size > len(self.buffer) or force_resize):
            self.buffer = None
        if self.buffer and new_size != len(self.buffer):
            self.views = None

    def calculate_size(self):
        return sum(sg(self.slice_count, self.batch_count) for sg in self.size_getters.values())

    def initialize_buffer(self, buffer=None):
        total_size = self.calculate_size()
        if buffer is None:
            self.buffer = pylstm_wrapper.BufferView(total_size)
        else:
            assert len(buffer) >= total_size
            self.buffer = buffer
        self.views = None

    def lay_out_views(self):
        assert self.buffer
        param_start = 0
        self.views = dict()
        for name, vfs in self.view_factories.items():
            param_size = self.size_getters[name](self.slice_count, self.batch_count)
            param_view = self.buffer[param_start : param_start + param_size]
            param_start += param_size
            self.views[name] = [vf(param_view, self.slice_count, self.batch_count) for vf in vfs]

    def get_buffer(self, name):
        if not self.buffer:
            self.initialize_buffer()
        if not self.views:
            self.lay_out_views()
        return self.views[name]