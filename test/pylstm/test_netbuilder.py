#!/usr/bin/python
# coding=utf-8

import unittest
from pylstm.netbuilder import NetworkBuilder, InvalidArchitectureError
from pylstm.layers import create_construction_layer


class Foo(): pass

FooLayer = create_construction_layer(Foo)


#noinspection PyStatementEffect
class ConstructionLayerTests(unittest.TestCase):
    def test_constructor(self):
        cl = FooLayer(7)
        self.assertEqual(cl.out_size, 7)
        self.assertEqual(cl.LayerType, Foo)

    def test_connecting_two_layers(self):
        cl1 = FooLayer(3)
        cl2 = FooLayer(4)
        cl1 >> cl2
        self.assertIn(cl1, cl2.sources)
        self.assertIn(cl2, cl1.targets)
        self.assertEqual(cl2.get_input_size(), 3)

    def test_connect_multiple_targets(self):
        l0, l1, l2, l3 = [FooLayer(10) for _ in range(4)]
        l0 >> l1
        l0 >> l2
        l0 >> l3
        self.assertIn(l1, l0.targets)
        self.assertIn(l2, l0.targets)
        self.assertIn(l3, l0.targets)

    def test_connect_multiple_sources(self):
        l0, l1, l2, l3 = [FooLayer(10) for _ in range(4)]
        l1 >> l0
        l2 >> l0
        l3 >> l0
        self.assertIn(l1, l0.sources)
        self.assertIn(l2, l0.sources)
        self.assertIn(l3, l0.sources)

    def test_get_depth_linear_architecture(self):
        layers = [FooLayer(10) for _ in range(5)]
        for i in range(4):
            layers[i] >> layers[i+1]
        layers[0].depth = 0
        self.assertListEqual([l.get_depth() for l in layers], list(range(5)))

    def test_traverse_targets_tree_linear_architecture(self):
        layers = [FooLayer(10) for _ in range(5)]
        for i in range(4):
            layers[i] >> layers[i+1]
        self.assertListEqual(list(layers[0].traverse_targets_tree()), layers)

    def test_get_depth_more_complicated(self):
        layers = [FooLayer(10) for _ in range(5)]
        l0, l1, l2, l3, l4 = layers
        l0 >> l1 >> l2 >> l3
        l0 >> l4 >> l3
        l0.depth = 0
        self.assertListEqual([l.get_depth() for l in layers], [0, 1, 2, 3, 1])

    def test_traverse_more_complicated(self):
        layers = [FooLayer(10) for _ in range(5)]
        l0, l1, l2, l3, l4 = layers
        l0 >> l1 >> l2 >> l3
        l0 >> l4 >> l3
        self.assertSetEqual(set(layers), set(l0.traverse_targets_tree()))

    def test_traverse_circle_raises_error(self):
        l0, l1, l2 = [FooLayer(10) for _ in range(3)]
        l0 >> l1 >> l2 >> l0
        try:
            list(l0.traverse_targets_tree())
            self.fail('Should have thrown')
        except InvalidArchitectureError:
            return
        except:
            self.fail('Wrong Exception')


class NetBuilderTests(unittest.TestCase):
    def setUp(self):
        self.netb = NetworkBuilder()

    def test_build_with_empty_network_raises(self):
        self.assertRaises(InvalidArchitectureError, self.netb.build)

#    def test_build_with_in_out_connection_works(self):
#        self.netb.input(3) >> self.netb.output
#        self.netb.build()

    def test_topological_sort(self):
        fl = FooLayer
        self.netb.input(3) >> fl(2) >> fl(2) >> fl(2) >> self.netb.output
        self.netb.input() >> fl(2) >> self.netb.output
        self.netb.input() >> self.netb.output
        layers = self.netb.get_sorted_layers()
        current_depth = 0
        for l in layers:
            self.assertGreaterEqual(l.get_depth(), current_depth)
            current_depth = l.get_depth()
            for s in l.sources:
                self.assertGreater(l.get_depth(), s.get_depth())
        self.assertEqual(len(layers), 6)
