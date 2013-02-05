#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import random
import numpy as np
import itertools

def binarize_sequence(seq, alphabet):
    result = np.zeros((len(seq), len(alphabet)))
    for i, s in enumerate(seq):
        result[i,alphabet.index(s)] = 1
    return result


def generate_memo_problem(pattern_length, alphabet_size, batch_size, length):
    """
    generate_memo_problem(5,  2, 32,         length) : generates the 5bit  problem
    generate_memo_problem(10, 5, batch_size, length) : generates the 20bit problem

    """
    assert alphabet_size >= 2, "need at least 2 characters (alphabet_size>=2)"
    assert length > 2*pattern_length + 2
    assert batch_size <= alphabet_size**pattern_length, "more batches than possible patterns"
    alphabet = range(alphabet_size)
    filler = [0]
    trigger = [1]
    inputs = []
    outputs = []
    mid_part_size = length - 2*pattern_length - 1
    if batch_size == alphabet_size**pattern_length:
        # make a batch for every possible pattern
        patterns = itertools.product(alphabet, repeat=pattern_length)
    else:
        patterns = [[random.choice(alphabet) for i in range(pattern_length)] for b in range(batch_size)]

    for pattern in patterns:
        pattern = list(pattern)
        in_seq = pattern + filler*mid_part_size + trigger + filler*pattern_length
        inputs.append(binarize_sequence(in_seq, alphabet))
        out_seq = filler * (length-pattern_length) + pattern
        outputs.append(binarize_sequence(out_seq, alphabet))

    return np.array(inputs).swapaxes(0,1), np.array(outputs).swapaxes(0,1)

