#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from copy import deepcopy

import unittest
from pylstm.structure.buffer_manager import BufferManager
from pylstm.structure.buffers2 import get_forward_closure


def generate_architecture(spec):
    """
    Generate a very basic architecture given a spec like this:
       "A>B B>C A>D D>C"
    """
    architecture = {}
    empty_layer = {
        'name': "",
        'targets': [],
        'sources': [],
        'size': 0
    }
    for con in spec.split():
        source, _, target = con.partition('>')
        if source not in architecture:
            architecture[source] = deepcopy(empty_layer)
            architecture[source]['name'] = source
        if target not in architecture:
            architecture[target] = deepcopy(empty_layer)
            architecture[target]['name'] = target
        architecture[source]['targets'].append(target)
        architecture[target]['sources'].append(source)
    return architecture


class BufferConstructionTest(unittest.TestCase):
    def setUp(self):
        self.forward_closure_tests = [
            # (architecture spec,
            #  {sources}, {sinks})
            ("A>B",
             {'A'}, {'B'}),
            ("A>B B>O I>A",
             {'A'}, {'B'}),
            ("A>B A>C",
             {'A'}, {'B', 'C'}),
            ("A>B C>B",
             {'A', 'C'}, {'B'}),
            ("A>B B>C A>C",
             {'A', 'B'}, {'B', 'C'}),
            ("A>B C>B C>D",
             {'A', 'C'}, {'B', 'D'}),
            ("A>B C>B C>D B>E D>E F>A G>C I>F I>G",
             {'A', 'C'}, {'B', 'D'}),
        ]

    def test_get_forward_closure(self):
        for spec, sources_expected, sinks_expected in self.forward_closure_tests:
            architecture = generate_architecture(spec)
            sources, sinks = get_forward_closure('A', architecture)
            self.assertSetEqual(sources, sources_expected)
            self.assertSetEqual(sinks, sinks_expected)


class BufferManagerTest(unittest.TestCase):

    def test_calculate_size_with_one_source_size_getter(self):
        bm = BufferManager()
        size_getter = lambda t, b : 10 * t + b
        view_factory = lambda x, t, b : x
        bm.add({'foo': (size_getter, view_factory)}, {})
        bm.set_dimensions(1, 1)
        self.assertEqual(bm.calculate_size(), 11)
        bm.set_dimensions(10, 5)
        self.assertEqual(bm.calculate_size(), 105)
        bm.set_dimensions(2, 3)
        self.assertEqual(bm.calculate_size(), 23)

    def test_calculate_size_with_one_sink_size_getter(self):
        bm = BufferManager()
        size_getter = lambda t, b : 10 * t + b
        view_factory = lambda x, t, b : x
        bm.add({}, {'foo': (size_getter, view_factory)})
        bm.set_dimensions(1, 1)
        self.assertEqual(bm.calculate_size(), 11)
        bm.set_dimensions(10, 5)
        self.assertEqual(bm.calculate_size(), 105)
        bm.set_dimensions(2, 3)
        self.assertEqual(bm.calculate_size(), 23)

    def test_calculate_size_with_one_sink_and_one_source_size_getter(self):
        bm = BufferManager()
        size_getter = lambda t, b : 10 * t + b
        view_factory = lambda x, t, b : x
        bm.add({'source': (size_getter, view_factory)}, {'sink': (size_getter, view_factory)})
        bm.set_dimensions(1, 1)
        self.assertEqual(bm.calculate_size(), 11)
        bm.set_dimensions(10, 5)
        self.assertEqual(bm.calculate_size(), 105)
        bm.set_dimensions(2, 3)
        self.assertEqual(bm.calculate_size(), 23)

    def test_calculate_size_with_multiple_size_getters(self):
        bm = BufferManager()
        sg1 = lambda t, b : 10*t + b
        sg2 = lambda t, b : 1000*t + 100*b
        sg3 = lambda t, b : 100000*t + 10000*b
        view_factory = lambda x, t, b : x
        bm.add({'foo1': (sg1, view_factory)}, {})
        bm.add({'foo2': (sg2, view_factory)}, {})
        bm.add({}, {'foo3': (sg3, view_factory)})
        bm.set_dimensions(1, 1)
        self.assertEqual(bm.calculate_size(), 111111)
        bm.set_dimensions(2, 3)
        self.assertEqual(bm.calculate_size(), 232323)

    def test_get_buffer_has_right_size(self):
        bm = BufferManager()
        sg1 = lambda t, b : 2 * t * b
        sg2 = lambda t, b : 3 * t * b
        view_factory = lambda x, t, b : x
        bm.add({'foo1': (sg1, view_factory), 'foo2': (sg2, view_factory)}, {})
        bm.set_dimensions(1, 1)
        self.assertEqual(len(bm.get_source_view('foo1')), 2)
        self.assertEqual(len(bm.get_source_view('foo2')), 3)
        bm.set_dimensions(2, 3)
        self.assertEqual(len(bm.get_source_view('foo1')), 12)
        self.assertEqual(len(bm.get_source_view('foo2')), 18)

if __name__ == "__main__":
    unittest.main()