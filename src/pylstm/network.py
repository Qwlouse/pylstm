#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import wrapper as pw

class Network(object):
    def __init__(self, layers, weight_manager, intern_manager, in_out_manager, intern_delta_manager, delta_manager):
        self.layers = layers
        self.weight_manager = weight_manager
        self.intern_manager = intern_manager
        self.in_out_manager = in_out_manager
        self.intern_delta_manager = intern_delta_manager
        self.delta_manager = delta_manager

    def get_param_size(self):
        """
        Returns the total size of all parameters.
        """
        return self.weight_manager.calculate_size()

    def get_input_size(self):
        return self.layers["Input"].get_output_size()

    def get_output_size(self):
        return self.layers["Output"].get_input_size()

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
        if isinstance(buffer, pw.BufferView):
            self.weight_manager.initialize_buffer(buffer)
        else:
            self.weight_manager.initialize_buffer(pw.BufferView(buffer))

    def forward_pass(self, input_buffer):
        # determine dimensions and set buffer managers accordingly
        t, b, f = input_buffer.shape
        assert f == self.layers.values()[0].get_output_size()
        self.intern_manager.set_dimensions(t, b)
        self.intern_delta_manager.set_dimensions(t, b)
        self.in_out_manager.set_dimensions(t, b)
        self.delta_manager.set_dimensions(t, b)
        # inject the input buffer
        self.in_out_manager.get_sink_view("Input").as_array()[:] = input_buffer
        # execute all the intermediate layers
        for n, l in self.layers.items()[1:-1]:
            param = self.weight_manager.get_source_view(n)
            internal = self.intern_manager.get_source_view(n)
            out = self.in_out_manager.get_sink_view(n)
            input_view = self.in_out_manager.get_source_view(n)
            l.forward(param, internal, input_view, out)
        # read the output buffer
        return self.in_out_manager.get_source_view("Output")

    def backward_pass(self, delta_buffer):
        # dimensions should already be set through forward_pass
        # inject delta_buffer
        out_view = self.delta_manager.get_source_view("Output").as_array()
        out_view[:] = delta_buffer
        # execute all the intermediate layers backwards
        for n, l in self.layers.items()[-2:0:-1]:
            param = self.weight_manager.get_source_view(n)
            internal = self.intern_manager.get_source_view(n)
            intern_delta = self.intern_delta_manager.get_source_view(n)
            out = self.in_out_manager.get_sink_view(n)
            delta_in = self.delta_manager.get_sink_view(n)
            delta_out = self.delta_manager.get_source_view(n)
            l.backward(param, internal, intern_delta, out, delta_in, delta_out)
            foo = 10
        # read the final delta buffer
        return self.delta_manager.get_sink_view("Input")
