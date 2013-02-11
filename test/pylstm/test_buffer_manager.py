#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals

import unittest
from pylstm.buffer_manager import BufferManager


class BufferManagerTest(unittest.TestCase):

    def test_calculate_size_with_one_source_size_getter(self):
        bm = BufferManager()
        size_getter = lambda t, b : 10 * t + b
        view_factory = lambda x, t, b : x
        bm.add({'foo': (size_getter, view_factory)}, {})
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
        self.assertEqual(bm.calculate_size(), 111111)
        bm.set_dimensions(2, 3)
        self.assertEqual(bm.calculate_size(), 232323)

    def test_get_buffer_has_right_size(self):
        bm = BufferManager()
        sg1 = lambda t, b : 2 * t + b
        sg2 = lambda t, b : t + 3*b
        view_factory = lambda x, t, b : x
        bm.add({'foo1': (sg1, view_factory), 'foo2': (sg2, view_factory)}, {})

        self.assertEqual(len(bm.get_source_view('foo1')), 3)
        self.assertEqual(len(bm.get_source_view('foo2')), 4)
        bm.set_dimensions(2, 3)
        self.assertEqual(len(bm.get_source_view('foo1')), 7)
        self.assertEqual(len(bm.get_source_view('foo2')), 11)

if __name__ == "__main__":
    unittest.main()