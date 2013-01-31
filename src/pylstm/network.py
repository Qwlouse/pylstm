#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

class Network(object):
    def __init__(self, layers, weight_manager, intern_manager, in_out_manager):
        self.layers = layers
        self.weight_manager = weight_manager
        self.intern_manager = intern_manager
        self.in_out_manager = in_out_manager

    def get_param_size(self):
        """
        Returns the total size of all parameters.
        """
        return self.weight_manager.calculate_size()

    def get_param_view_for(self, name):
        return self.weight_manager.get_buffer(name)[0]

    def get_intern_view_for(self, name):
        return self.intern_manager.get_buffer(name)[0]

    def get_output_view_for(self, name):
        return self.in_out_manager.get_buffer(name)[0]

    def __getitem__(self, item):
        """
        Get the layer with the given name.
        """
        return self.layers[item]

    def set_param_buffer(self, buffer):
        """
        Set the parameter buffer that holds all the weights.
        """
        self.weight_manager.initialize_buffer(buffer)

    def forward_pass(self, input_buffer):
        # determine dimensions and set buffer managers accordingly
        f = input_buffer.get_feature_size()
        t = input_buffer.get_time_size()
        b = input_buffer.get_batch_size()
        assert f == self.layers.values()[0].get_output_size()
        self.intern_manager.set_dimensions(t, b)
        self.in_out_manager.set_dimensions(t, b)
        # inject the input buffer
        self.in_out_manager.get_sink_view("Input").as_array()[:] = input_buffer.as_array()
        # execute all the intermediate layers
        for n, l in self.layers.items()[1:-1]:
            param = self.weight_manager.get_source_view(n)
            internal = self.intern_manager.get_source_view(n)
            out = self.in_out_manager.get_sink_view(n)
            input_view = self.in_out_manager.get_source_view(n)
            l.forward(param, internal, input_view, out)
        # read the output buffer
        return self.in_out_manager.get_source_view("Output")