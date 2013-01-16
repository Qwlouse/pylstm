#!/usr/bin/python
# coding=utf-8
import pylstm_wrapper
from collections import OrderedDict

class InvalidArchitectureError(RuntimeError):
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

DummyLayer = create_ConstructionLayer(None)

class Network(object):
    def __init__(self, layers, views, param_buffer, internal_buffers, forward_buffers):
        self.layers = layers
        self.views = views
        self.param_buffer = param_buffer
        self.internal_buffers = internal_buffers
        self.forward_buffers = forward_buffers
        assert len(layers) == len(views) == len(internal_buffers)

    def forward_pass(self, X):
        self.forward_buffers[0].assign(X)
        for l, v in zip(self.layers, self.views):
            l.forward(**v)
        Y = self.forward_buffers[-1]
        return Y

class BufferManager(object):
    def __init__(self):
        self.size_getters = {}
        self.view_factories = {}
        self.buffer = None
        self.views = {}

    def add(self, name, size_getter, view_factories):
        self.buffer = None
        self.size_getters[name] = size_getter
        self.view_factories[name] = view_factories

    def calculate_size(self, slice_count=1, batch_count=1):
        return sum(sg(slice_count, batch_count) for sg in self.size_getters)

    def initialize_buffer(self, slice_count=1, batch_count=1):
        total_size = self.calculate_size(slice_count, batch_count)
        self.buffer = pylstm_wrapper.MatrixView(
            pylstm_wrapper.MatrixCPU(total_size, 1, 1))

    def lay_out_views(self, slice_count=1, batch_count=1):
        param_start = 0
        for name, vfs in self.view_factories.items():
            param_size = self.size_getters[name](slice_count, batch_count)
            param_view = self.buffer.slice(param_start, param_start + param_size)
            param_start += param_size
            self.views[name] = [vf(param_view, slice_count, batch_count) for vf in vfs]

    def get_buffer(self, name, slice_count=1, batch_count=1):
        if not self.buffer:
            self.initialize_buffer(slice_count, batch_count)
            self.lay_out_views(slice_count, batch_count)
        else:
            # check sizes
            new_size = self.calculate_size(slice_count, batch_count)
            if new_size > self.buffer.get_slice_count():
                self.initialize_buffer(slice_count, batch_count)
            if new_size < self.buffer.get_slice_count():
                self.lay_out_views(slice_count, batch_count)
        return self.views[name]



class NetworkBuilder():
    def __init__(self):
        self.input_layer = None
        self.output = DummyLayer(0)

    def input(self, size=None):
        if size :
            self.input_layer = DummyLayer(size)
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
            new_lset = {s for s in l.sources for l in rset}
            new_rset = {t for t in l.targets for l in lset}
            if len(new_lset) > len(lset) or\
               len(new_rset) > len(rset)    :
                growing = True
                lset = new_lset
                rset = new_rset
        return lset, rset

    def create_buffer(self, size):
        return pylstm_wrapper.MatrixView(pylstm_wrapper.MatrixCPU(1, 1, size))

    def build(self):
        """
        Turn a _linear_ network graph into a Network object.
        ATM More complicated graph layouts will give wrong results or fail.
        """
        cLayers = self.get_sorted_layers()
        assert cLayers[0] is self.input_layer
        assert cLayers[-1] is self.output
        layers = [cl.instantiate() for cl in cLayers[1:-1]] # without in and out layer

        total_param_size = sum(l.get_param_size() for l in layers) or 1
        param_buffer = self.create_buffer(total_param_size)
        internal_buffers = []
        in_buffer = self.create_buffer(self.input_layer.out_size)
        forward_buffers = [in_buffer]
        current_in_buffer = in_buffer
        views = []
        param_start = 0
        for layer in layers:
            param_size = layer.get_param_size()
            param_view = param_buffer.slice(param_start, param_start + param_size)
            param_start += param_size

            internal_buffer = self.create_buffer(layer.get_internal_state_size())
            internal_buffers.append(internal_buffer)

            out_buffer = self.create_buffer(layer.get_output_size())
            forward_buffers.append(out_buffer)

            views.append(dict(
                input=layer.create_input_view(current_in_buffer),
                param=layer.create_param_view(param_view),
                internal=layer.create_internal_view(internal_buffer),
                output=layer.create_output_view(out_buffer)
            ))
            current_in_buffer = out_buffer
        return Network(layers, views, param_buffer, internal_buffers, forward_buffers)


    def get_named_layers(self):
        # instantiate all the layers with names
        cLayers = self.get_sorted_layers()
        assert cLayers[0] is self.input_layer
        assert cLayers[-1] is self.output
        layers = OrderedDict()
        for l in cLayers[1:-1]:
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
        return layers, cLayers

    def build2(self):
        layers, cLayers = self.get_named_layers()

        weight_manager = BufferManager()
        for name, l in layers.items():
            weight_manager.add(name, l.get_param_size, [l.create_param_view])

        intern_manager = BufferManager()
        for name, l in layers.items():
            intern_manager.add(name, l.get_internal_state_size, [l.create_internal_view])

        output_manager = BufferManager()
        for layer in cLayers.items():
            lset, rset = self.get_forward_closure(layer)
            assert len(lset)==len(rset)==1, "Complicated Architectures not supported yet"
            name = lset.pop().name
            l = layers[name]
            r = layers[rset.pop().name]
            output_manager.add(name, l.get_output_buffer_size, [l.create_output_view, r.create_input_view])



        # WIP...


################################################################################

LstmLayer = create_ConstructionLayer(pylstm_wrapper.LstmLayer)