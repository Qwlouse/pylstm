#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
import random
import collections
from .preprocessing import binarize_sequence


alphabet = ["A", "B", "C"]


def uniform(min, max):
    np.random.randint(min, max + 1)


def generate_sequences(alphabet, length_func, seq_count):
    seqs = set()
    while len(seqs) < seq_count:
        length = length_func if isinstance(length_func, int) else length_func()
        seq = tuple(random.choice(alphabet) for l in range(length))
        if seq not in seqs:
            seqs.add(seq)
    return list(seqs)


def generate_sequence_hierarchy(alph, hierarchy_tuples):
    for (lenf, count) in hierarchy_tuples:
        alph = generate_sequences(alph, lenf, count)
    return alph


def recursive_flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in recursive_flatten(el):
                yield sub
        else:
            yield el


def generate_flattened_binarized_sequence_hierarchy(alph, hierarchy_tuples):
    seq = generate_sequence_hierarchy(alph, hierarchy_tuples)
    print(seq)
    fseq = list(recursive_flatten(seq))
    print(fseq)
    return np.array(binarize_sequence(fseq))


def generate_hierarchical_1step_prediction_problem(alphabet_size, nr_batches, length, seqs_per_level, levels):
    dataset = []
    alphabet = range(alphabet_size)
    base_sequences = generate_sequence_hierarchy(alphabet, [(length, seqs_per_level)] * levels)
    for batch in range(nr_batches):
        seq = binarize_sequence(list(recursive_flatten(generate_sequence_hierarchy(base_sequences, [(length, 5)]))))
        dataset.append(seq)
    dataset = np.array(dataset).swapaxes(0, 1)
    X = dataset[:-1, :, :]
    T = dataset[1:, :, :]

    return X, T


def main():
    print(generate_hierarchical_1step_prediction_problem(3, 1, 3, 3, 1))

if __name__ == '__main__':
    main()

