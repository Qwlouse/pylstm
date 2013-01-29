#!/usr/bin/python
# coding=utf-8
import wrapper
from collections import OrderedDict
from buffer_manager import BufferManager
from layers import DummyLayer, InvalidArchitectureError

class Network(object):
    def __init__(self, layers, weight_manager, intern_manager, output_manager):
        self.layers = layers
        self.weight_manager = weight_manager
        self.intern_manager = intern_manager
        self.output_manager = output_manager

    def get_param_size(self):
        """
        Returns the total size of all parameters.
        """
        return self.weight_manager.calculate_size()

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
        f = input_buffer.get_feature_size()
        t = input_buffer.get_time_size()
        b = input_buffer.get_batch_size()
        assert f == self.layers.values()[0].get_input_size()
        self.intern_manager.set_dimensions(t, b)
        self.output_manager.set_dimensions(t, b)
        input_view = self.layers.values()[0].create_input_view(input_buffer, t, b)
        for n, l in self.layers.items()[:-1]:
            param = self.weight_manager.get_buffer(n)[0]
            internal = self.intern_manager.get_buffer(n)[0]
            out = self.output_manager.get_buffer(n)
            l.forward(param, internal, input_view, out[0])
            input_view = out[1]
        return input_view

class NetworkBuilder():
    def __init__(self):
        self.input_layer = None
        self.output = DummyLayer(0, "Output")

    def input(self, size=None):
        if size :
            self.input_layer = DummyLayer(size, "Input")
        return self.input_layer

    def get_sorted_layers(self):
        if not self.input_layer:
            raise InvalidArchitectureError("Empty")
        # gather all the layers
        layers = set(self.input_layer.traverse_targets_tree())
        # sort them by depth
        self.input_layer.depth = 0
        return sorted(layers, key=lambda l: l.get_depth())

    def get_forward_closure(self, layer):
        """
        for a given layer return two sets of layers such that:
          * the given layer is in the first set
          * the second set contains all the target layers of the first set
          * the first set contains all the source layers of the second set
        """
        lset = {layer}
        rset = set(layer.targets)
        growing = True
        while growing:
            growing = False
            new_lset = {s for l in rset for s in l.sources}
            new_rset = {t for l in lset for t in l.targets}
            if len(new_lset) > len(lset) or\
               len(new_rset) > len(rset)    :
                growing = True
                lset = new_lset
                rset = new_rset
        return lset, rset

    def create_buffer(self, size):
        return wrapper.BufferView(1, 1, size)

    def get_named_layers(self):
        # instantiate all the layers with names
        cLayers = self.get_sorted_layers()
        assert cLayers[0] is self.input_layer
        assert cLayers[-1] is self.output
        layers = OrderedDict()
        for l in cLayers[1:]:
            layer = l.instantiate()
            name = l.get_name()
            # ensure unique name
            if name in layers:
                basename = name
                idx = 1
                while name in layers:
                    name = basename + "_%d"%idx
                    idx += 1
            l.name = name
            layers[l.name] = layer
        return layers, cLayers[1:]

    def build(self):
        layers, cLayers = self.get_named_layers()

        weight_manager = BufferManager()
        for name, l in layers.items():
            weight_manager.add(name, l.get_param_size, [l.create_param_view])

        intern_manager = BufferManager()
        for name, l in layers.items():
            intern_manager.add(name, l.get_internal_state_size, [l.create_internal_view])

        output_manager = BufferManager()
        for layer in cLayers[:-1]:
            lset, rset = self.get_forward_closure(layer)
            assert len(lset)==len(rset)==1, "Complicated Architectures not supported yet"
            name = lset.pop().name
            l = layers[name]
            r = layers[rset.pop().name]
            output_manager.add(name, l.get_output_size, [l.create_output_view, r.create_input_view])

        net = Network(layers, weight_manager, intern_manager, output_manager)
        return net

