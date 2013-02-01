#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import wrapper

class InvalidArchitectureError(RuntimeError):
    pass

class Layer(object):
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size

    def get_output_size(self):
        return self.out_size

    def get_input_size(self):
        return self.in_size

    def get_input_buffer_size(self, time_length=1, batch_size=1):
        return self.in_size * time_length * batch_size

    def get_output_buffer_size(self, time_length=1, batch_size=1):
        return self.out_size * time_length * batch_size

    def get_param_size(self, time_length=1, batch_size=1):
        return 0

    def get_internal_state_size(self, time_length, batch_size):
        return 0

    def get_internal_error_state_size(self, time_length, batch_size1):
        return 0

    def create_input_view(self, input_buffer, time_length, batch_size):
        assert len(input_buffer) == self.get_input_buffer_size(time_length, batch_size)
        return input_buffer.reshape(time_length, batch_size, self.in_size)

    def create_output_view(self, output_buffer, time_length, batch_size):
        assert len(output_buffer) == self.get_output_buffer_size(time_length, batch_size)
        return output_buffer.reshape(time_length, batch_size, self.out_size)

    def create_param_view(self, param_buffer, time_length=1, batch_size=1):
        return None

    def create_internal_view(self, internal_buffer, time_length, batch_size):
        return None

    def create_internal_error_view(self, error_buffer, time_length, batch_size):
        return None

    def forward(self, param, internal, input, output):
        pass

    def backward(self, param, internal, err, output, in_deltas, out_deltas):
        pass


def create_ConstructionLayer(LayerType):
    class ConstructionLayer(object):
        def __init__(self, out_size, name = None, **layer_kwargs):
            self.out_size = out_size
            self.name = name
            self.targets = []
            self.sources = []
            self.LayerType = LayerType
            self.layer_kwargs = layer_kwargs
            self.depth = None
            self.traversing = False

        def instantiate(self):
            return self.LayerType(self.get_input_size(),
                self.out_size, **self.layer_kwargs)

        def get_input_size(self):
            return sum(s.out_size for s in self.sources)

        def get_depth(self):
            if self.depth is None:
                self.depth = float('inf') # marker for "get_depth" in progress
                self.depth = max(s.get_depth() for s in self.sources) + 1
            return self.depth

        def traverse_targets_tree(self):
            if self.traversing:
                raise InvalidArchitectureError("Circle in Network")
            self.traversing = True
            yield self
            for target in self.targets:
                for t in target.traverse_targets_tree():
                    yield t
            self.traversing = False

        def _add_source(self, other):
            self.sources.append(other)

        def __rshift__(self, other):
            self.targets.append(other)
            other._add_source(self)
            return other

        def get_name(self):
            return self.name or self.LayerType.__name__

    return ConstructionLayer


################################################################################

DummyLayer = create_ConstructionLayer(Layer)
LstmLayer = create_ConstructionLayer(wrapper.LstmLayer)