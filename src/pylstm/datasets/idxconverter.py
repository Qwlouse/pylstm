#!/usr/bin/python
# coding=utf-8
"""
Convert IDX files to numpy and back.
http://yann.lecun.com/exdb/mnist/  # bottom of the page
"""
from __future__ import division, print_function, unicode_literals
import numpy as np

__all__ = ['open_idx_file', 'write_idx_file']

TYPE_DICT = {
    0x8: np.uint8,
    0x9: np.int8,
    0xB: np.int16,
    0xC: np.int32,
    0xD: np.float32,
    0xE: np.float64
}
REVERSE_TYPE_DICT = {
    np.zeros(1, dtype=np.uint8).dtype: b"\x08",
    np.zeros(1, dtype=np.int8).dtype: b"\x09",
    np.zeros(1, dtype=np.int16).dtype: b"\x0B",
    np.zeros(1, dtype=np.int32).dtype: b"\x0C",
    np.zeros(1, dtype=np.uint32).dtype: b"\x0C",
    np.zeros(1, dtype=np.float32).dtype: b"\x0D",
    np.zeros(1, dtype=np.float64).dtype: b"\x0E"
}


def open_idx_file(filename, byteswap=True, reverse_header=False):
    """
    Open idx file with given filename and returns a numpy array.
    @param filename: path to the idx file
    @param byteswap: if True then data is read as big-endian
    @param reverse_header: if True the header is read in reverse
                           (some of Dan's files use this)

    @returns ndarray with correct datatype and shape
    """
    with open(filename, 'rb') as f:
        if not reverse_header:
            zero = np.fromfile(f, np.int16, 1).byteswap()[0]
            assert zero == 0, \
                "File should start with two zero-bytes but was %s." % hex(zero)
            type_code = np.fromfile(f, np.uint8, 1)[0]
            assert type_code in TYPE_DICT, \
                "Invalid type code %s." % hex(type_code)
            dim_count = np.fromfile(f, np.uint8, 1)[0]
            shape = np.fromfile(f, np.int32, dim_count).byteswap()
        else:
            dim_count = np.fromfile(f, np.uint8, 1)[0]
            print("dim_count", dim_count)
            type_code = np.fromfile(f, np.uint8, 1)[0]
            print("type_code", type_code)
            assert type_code in TYPE_DICT, \
                "Invalid type code %s." % hex(type_code)
            zero = np.fromfile(f, np.int16, 1).byteswap()[0]
            assert zero == 0, \
                "File should start with two zero-bytes but was %s." % hex(zero)
            shape = np.fromfile(f, np.int32, dim_count)

        print("shape", shape)
        size = reduce(np.multiply, shape)
        if byteswap:
            return np.fromfile(f, TYPE_DICT[type_code], size).byteswap()\
                .reshape(shape)
        else:
            return np.fromfile(f, TYPE_DICT[type_code], size).reshape(shape)


def write_idx_file(filename, A, shape=None, byteswap=True):
    """
    Writes a numpy array to a idx file.
    @param filename: name of target file
    @param A: the ndarray that will be written
    @param shape: if given then this is written to file instead of A.shape
    @param byteswap: if True the data is written as big-endian
    """
    with open(filename, 'wb') as f:
        f.write(b"\x00\x00")  # zeros
        f.write(REVERSE_TYPE_DICT[A.dtype])  # type
        f.write(b"%c" % len(A.shape))  # nr of dims
        if shape is None:
            shape = A.shape
        np.array(shape, dtype=np.int32).byteswap(True).tofile(f)  # shape
        if byteswap:
            A.byteswap(False).tofile(f)  # numbers
        else:
            A.tofile(f)  # numbers
