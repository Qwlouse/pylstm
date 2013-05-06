#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import random
import numpy as np
import itertools
import collections


def binarize_sequence(seq, alphabet = None):
    if alphabet is None:
        alphabet = np.lib.arraysetops.unique(seq)
    else:
        alphabet = np.array(alphabet)
    result = np.zeros((len(seq), len(alphabet)))
    for i, s in enumerate(seq):
        index = np.where(alphabet == s)[0][0]
        result[i, index] = 1
    return result


def generate_memo_task(pattern_length, alphabet_size, batch_size, length):
    """
    generate_memo_task(5,  2, 32,         length) : generates 5bit  problem
    generate_memo_task(10, 5, batch_size, length) : generates 20bit problem

    """
    assert alphabet_size >= 2, "need at least 2 characters (alphabet_size>=2)"
    assert length > 2 * pattern_length + 2
    assert batch_size <= alphabet_size ** pattern_length, \
        "more batches than possible patterns"
    alphabet = range(alphabet_size)
    filler = [0]
    trigger = [1]
    inputs = []
    outputs = []
    mid_part_size = length - 2 * pattern_length - 1
    if batch_size == alphabet_size ** pattern_length:
        # make a batch for every possible pattern
        patterns = itertools.product(alphabet, repeat=pattern_length)
    else:
        patterns = [[random.choice(alphabet) for _ in range(pattern_length)]
                    for _ in range(batch_size)]

    for pattern in patterns:
        pattern = list(pattern)
        in_seq = pattern + filler * mid_part_size + trigger + \
            filler * pattern_length
        inputs.append(binarize_sequence(in_seq, alphabet))
        out_seq = filler * (length - pattern_length) + pattern
        outputs.append(binarize_sequence(out_seq, alphabet))

    return np.array(inputs).swapaxes(0, 1), np.array(outputs).swapaxes(0, 1)


def generate_5bit_task(total_length):
    return generate_memo_task(5,  2, 32, total_length)


def generate_20bit_task(total_length, batch_size=1000):
    return generate_memo_task(10,  5, batch_size, total_length)


def generate_math_task(T0, batch_size, operation=np.add,
                          rnd=np.random.RandomState()):
    """
    In this task there are two input channels (L = 2). The first channel
    receives a stream u1(n) of random reals sampled uniformly from [0, 1].
    The second channel receives zero input u2(n) = 0 at all times except at two
    timesteps n1 < n2 when u2(n1) = u2(n2) = 1. The objective is that at the
    end of a run (much later than n1 or n2), the network should output the
    normalized sum (u1 (n1) + u1 (n2)) / 2. An additional difficulty is that the
    length of input sequences varies randomly.
    """
    Ti = rnd.randint(T0, 1.1 * T0, batch_size)
    n1 = rnd.randint(1, 0.1 * T0, batch_size)
    n2 = rnd.randint(0.1 * T0 + 1, 0.5 * T0, batch_size)
    u1 = rnd.random_sample((T0*1.1, batch_size))
    u2 = np.zeros_like(u1)
    T = np.zeros_like(u1)
    for b in range(batch_size):
        u2[n1[b], b] = 1.
        u2[n2[b], b] = 1.
        T[Ti[b]-1, b] = operation(u1[n1[b], b], u1[n2[b], b])
    X = np.dstack((u1, u2))
    M = np.zeros_like(T)
    M[Ti-1] = 1.
    return X, T, M


def generate_addition_task(T0, batch_size, rnd=np.random.RandomState()):
    return generate_math_task(T0, batch_size, operation=np.add, rnd=rnd)


def generate_multiplication_task(T0, batch_size, rnd=np.random.RandomState()):
    return generate_math_task(T0, batch_size, operation=np.multiply, rnd=rnd)





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

