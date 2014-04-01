#!/usr/bin/python
# coding=utf-8

import unittest
from pylstm.structure import build_net, InvalidArchitectureError
from pylstm.structure.layers import InputLayer, create_construction_layer
from pylstm.structure.netbuilder import create_architecture_from_layers


class Foo(): pass

FooLayer = create_construction_layer('Foo')


#noinspection PyStatementEffect
class ConstructionLayerTests(unittest.TestCase):
    def test_constructor(self):
        cl = FooLayer(7)
        self.assertEqual(cl.out_size, 7)
        self.assertEqual(cl.layer_type, 'Foo')

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

    def test_traverse_targets_tree_linear_architecture(self):
        layers = [FooLayer(10) for _ in range(5)]
        for i in range(4):
            layers[i] >> layers[i+1]
        self.assertListEqual(list(layers[0].traverse_targets_tree()), layers)

    def test_traverse_more_complicated(self):
        layers = [FooLayer(10) for _ in range(5)]
        l0, l1, l2, l3, l4 = layers
        l0 >> l1 >> l2 >> l3
        l0 >> l4 >> l3
        self.assertSetEqual(set(layers), set(l0.traverse_targets_tree()))

    def test_traverse_circle_raises_error(self):
        l0, l1, l2 = [FooLayer(10) for _ in range(3)]
        l0 >> l1 >> l2 >> l0
        with self.assertRaises(InvalidArchitectureError):
            list(l0.traverse_targets_tree())

    def test_building_forked_architectures(self):
        # This test passes if building forked networks succeeds
        i = InputLayer(10)
        l1, l2, l3 = [FooLayer(10) for _ in range(3)]
        i >> l1 >> l3
        i >> l2 >> l3
        create_architecture_from_layers(i)
