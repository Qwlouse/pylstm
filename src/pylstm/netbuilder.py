#!/usr/bin/python
# coding=utf-8

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

    def build(self):
        if not self.output.sources:
            raise InvalidArchitectureError()
        # WIP...


################################################################################

#LstmLayer = create_ConstructionLayer(layers.Lstm)