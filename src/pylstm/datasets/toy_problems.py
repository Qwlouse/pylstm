#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import random
import numpy as np
import itertools
from .preprocessing import binarize_sequence


def generate_memo_task(pattern_length, alphabet_size, batch_size, length,
                       filler=None, trigger=None):
    """
    generate_memo_task(5,  2, 32,         length) : generates 5bit  problem
    generate_memo_task(10, 5, batch_size, length) : generates 20bit problem

    """
    assert alphabet_size >= 2, "need at least 2 characters (alphabet_size>=2)"
    assert length > 2 * pattern_length + 2
    assert batch_size <= alphabet_size ** pattern_length, \
        "more batches than possible patterns"

    filler = [alphabet_size] if filler is None else [filler]
    trigger = [alphabet_size + 1] if trigger is None else [trigger]
    alphabet = range(alphabet_size)
    full_alphabet = set(alphabet)
    full_alphabet.add(filler[0])
    full_alphabet.add(trigger[0])
    full_alphabet = sorted(list(full_alphabet))

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
        inputs.append(binarize_sequence(in_seq, full_alphabet))
        out_seq = filler * (length - pattern_length) + pattern
        outputs.append(binarize_sequence(out_seq, full_alphabet))

    return np.array(inputs).swapaxes(0, 1), np.array(outputs).swapaxes(0, 1)


def generate_5bit_memory_task(total_length):
    return generate_memo_task(5,  2, 32, total_length)


def generate_20bit_memory_task(total_length, batch_size=1000):
    return generate_memo_task(10,  5, batch_size, total_length)


def generate_5bit_memory_task_compact(total_length):
    return generate_memo_task(5,  2, 32, total_length, filler=0, trigger=1)


def generate_20bit_memory_task_compact(total_length, batch_size=1000):
    return generate_memo_task(10,  5, batch_size, total_length,
                              filler=0, trigger=1)


def generate_math_task(T0, batch_size, operation=np.add, input_binary=False,
                       rnd=np.random.RandomState()):
    """
    In this task there are two input channels (L = 2). The first channel
    receives a stream u1(n) of random (reals # if not input_binary) sampled
    (uniformly) from [0, 1].
    The second channel receives zero input u2(n) = 0 at all times except at two
    timesteps n1 < n2 when u2(n1) = u2(n2) = 1. The objective is that at the
    end of a run (much later than n1 or n2), the network should output the
    the given binary operation on u1(n1) and u1(n2). An additional difficulty
    is that the length of input sequences varies randomly.
    """
    Ti = rnd.randint(T0, 1.1 * T0, batch_size)
    n1 = rnd.randint(1, 0.1 * T0, batch_size)
    n2 = rnd.randint(0.1 * T0 + 1, 0.5 * T0, batch_size)
    u1 = rnd.random_sample((T0*1.1, batch_size))
    if input_binary:
        u1 = np.round(u1)
    u2 = np.zeros_like(u1)
    T = np.zeros_like(u1).reshape(-1, batch_size, 1)
    M = np.zeros_like(T)
    for b in range(batch_size):
        u2[n1[b], b] = 1.
        u2[n2[b], b] = 1.
        T[Ti[b]-1, b] = operation(u1[n1[b], b], u1[n2[b], b])
        M[Ti[b]-1, b] = 1.
    X = np.dstack((u1, u2))
    return X, T, M


def generate_addition_task(T0, batch_size, rnd=np.random.RandomState()):
    return generate_math_task(T0, batch_size, rnd=rnd,
                              operation=lambda a, b: np.add(a, b)/2)


def generate_multiplication_task(T0, batch_size, rnd=np.random.RandomState()):
    return generate_math_task(T0, batch_size, operation=np.multiply, rnd=rnd)


def generate_xor_task(T0, batch_size, rnd=np.random.RandomState()):
    return generate_math_task(T0, batch_size, operation=np.logical_xor,
                              input_binary=True, rnd=rnd)


def generate_temporal_order_task(length, batch_size,
                                 symbol_occurences=((.1, .2), (.5, .6)),
                                 rnd=np.random.RandomState()):
    X = np.zeros((length, batch_size, 6))
    T = np.zeros((length, batch_size, 2 ** len(symbol_occurences)))
    M = np.zeros((length, batch_size, 1))
    # distractor pattern
    for t in range(length):
        for b in range(batch_size):
            X[t, b, rnd.randint(0, 4)] = 1.

    # the symbols
    for b in range(batch_size):
        out = 0
        for T_range in symbol_occurences:
            Ti = rnd.randint(length*T_range[0], length*T_range[1])
            X[Ti, b, :] = 0
            A_or_B = rnd.randint(0, 2)
            X[Ti, b, A_or_B + 4] = 1.
            out *= 2
            out += A_or_B
        T[-1, b, out] = 1
    M[-1] = 1
    return X, T, M


def generate_2_symbol_temporal_order_task(length, batch_size,
                                          rnd=np.random.RandomState()):
    generate_temporal_order_task(length, batch_size, rnd=rnd,
                                 symbol_occurences=((.1, .2),
                                                    (.5, .6)))


def generate_3_symbol_temporal_order_task(length, batch_size,
                                          rnd=np.random.RandomState()):
    generate_temporal_order_task(length, batch_size, rnd=rnd,
                                 symbol_occurences=((.1, .2),
                                                    (.3, .4),
                                                    (.6, .7)))


def generate_random_permutation_task(length, batch_size, rnd=np.random.RandomState()):
    outputs = []

    for b in range(batch_size):
        seq = rnd.randint(2, 100, length)
        seq[0] = rnd.randint(0, 2)
        outputs.append(binarize_sequence(seq, range(100)))

    X = np.array(outputs).swapaxes(0, 1)
    T = np.zeros_like(X)
    T[0:-1, :, :] = X[1:, :, :]
    T[-1, :, :] = X[0, :, :]
    M = np.zeros((length, batch_size, 1))
    M[-1, :, :] = 1
    return X, T, M