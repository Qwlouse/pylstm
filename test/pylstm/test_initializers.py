#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import unittest

from pylstm.structure.netbuilder import build_net
from pylstm.regularization.initializer import (_create_initializer_from_description, Gaussian,
                                               SparseInputs, Initializer)
from pylstm.structure.layers import InputLayer, ForwardLayer


class InitializerTest(unittest.TestCase):
    def setUp(self):
        self.net = build_net(InputLayer(1) >> ForwardLayer())

    def test_create_initializer_from_plain_number(self):
        self.assertEqual(_create_initializer_from_description(0.0), 0.0)
        self.assertEqual(_create_initializer_from_description(1.0), 1.0)
        self.assertEqual(_create_initializer_from_description(5), 5)

    def test_create_initializer_from_list(self):
        self.assertListEqual(_create_initializer_from_description([1]), [1])
        self.assertListEqual(_create_initializer_from_description([1, 2, 3]), [1, 2, 3])
        self.assertListEqual(_create_initializer_from_description([1.0, -1.0]), [1.0, -1.0])

    def test_create_gaussian_initializer_from_dict(self):
        description = {
            '$type': 'Gaussian',
            'std': 23.0,
            'mean': 7.0
        }
        init = _create_initializer_from_description(description)
        self.assertIsInstance(init, Gaussian)
        self.assertEqual(init.mean, 7.0)
        self.assertEqual(init.std, 23.0)

    def test_get_description_of_gaussian_initializer(self):
        init = Gaussian(mean=9.0, std=42.0)
        description = init.__get_description__()
        self.assertEqual(description['$type'], 'Gaussian')
        self.assertEqual(description['mean'], 9.0)
        self.assertEqual(description['std'], 42.0)

    def test_sparse_inputs_initalizer_description(self):
        init = SparseInputs(Gaussian(mean=3.0, std=4.0), 7)
        description = init.__get_description__()
        self.assertEqual(description['$type'], 'SparseInputs')
        self.assertEqual(description['connections'], 7)
        self.assertDictEqual(description['init'], {'$type': 'Gaussian',
                                                   'std': 4.0,
                                                   'mean': 3.0})

    def test_sparse_inputs_initalizer_from_description(self):
        description = {
            '$type': 'SparseInputs',
            'connections': 69,
            'init': {'$type': 'Gaussian',
                     'std': 23.0,
                     'mean': 7.0}
        }
        init = _create_initializer_from_description(description)
        self.assertIsInstance(init, SparseInputs)
        self.assertEqual(init.connections, 69)
        self.assertIsInstance(init.init, Gaussian)
        self.assertEqual(init.init.std, 23.0)
        self.assertEqual(init.init.mean, 7.0)

    def test_custom_initializer(self):
        class Custom(Initializer):
            def __init__(self, foo):
                Initializer.__init__(self)
                self.foo = foo

        c = Custom('bar')
        descr = c.__get_description__()
        self.assertDictEqual(descr, {'$type': 'Custom',
                                     'foo': 'bar'})

        c2 = _create_initializer_from_description(descr)
        self.assertNotEqual(c, c2)
        self.assertIsInstance(c2, Custom)
        self.assertEqual(c2.foo, 'bar')


