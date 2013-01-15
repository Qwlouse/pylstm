#!/usr/bin/python
# coding=utf-8
import pylstm_wrapper

class InvalidArchitectureError(RuntimeError):
    pass

def create_ConstructionLayer(LayerType):
    class ConstructionLayer(object):
        def __init__(self, out_size, **layer_kwargs):
            self.out_size = out_size
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

class NetworkBuilder():
    def __init__(self):
        self.input_layer = None
        self.output = DummyLayer(0)

    def input(self, size=None):
        if size :
            self.input_layer = DummyLayer(size)
        return self.input_layer

    def get_sorted_layers(self):
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

        total_param_size = sum(l.get_param_size() for l in layers)
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


        # WIP...


################################################################################

LstmLayer = create_ConstructionLayer(pylstm_wrapper.LstmLayer)