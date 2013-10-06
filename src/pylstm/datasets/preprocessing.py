#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from random import shuffle


def binarize_sequence(seq, alphabet=None):
    """
    Binarize a sequence given as a list or a 1d array. Returns a 2d one-hot
    representation of that sequence.
    You can specify the alphabet for binarization yourself. Otherwise it will be
    the symbols from the passed in sequence.
    """
    if alphabet is None:
        alphabet = np.lib.arraysetops.unique(seq)
    else:
        alphabet = np.array(alphabet)
    result = np.zeros((len(seq), len(alphabet)))
    for i, s in enumerate(seq):
        index = np.where(alphabet == s)[0][0]
        result[i, index] = 1
    return result


def binarize_array(A, alphabet=None):
    """
    Binarize a batch of sequences given as a 2d or 3d array. Dimensions have to
    be (time, seq_nr[, 1]). Returns a 3d one-hot representation of all those
    sequences.
    You can specify the alphabet for binarization yourself. Otherwise it will be
    the symbols from the passed in sequences.
    """
    if A.shape == 3:
        assert A.shape[2] == 1
        A = A[:, :, 0]
    if alphabet is None:
        alphabet = list(np.unique(A))
    else:
        alphabet = list(alphabet)
    result = np.zeros((A.shape[0], A.shape[1], len(alphabet)))
    for seq_nr in range(A.shape[1]):
        for t in range(A.shape[0]):
            index = alphabet.index(A[t, seq_nr])
            result[t, seq_nr, index] = 1
    return result


def get_mean_masked(X, M):
    """
    Get the mean values for every feature in the batch of sequences X by
    considering only masked-in entries.
    :param X: Batch of sequences. shape = (time, sample, feature)
    :param M: Mask for the sequences. shape = (time, sample, 1)
    :returns: mean value for each feature. shape = (features, )
    """
    return X.reshape(-1, X.shape[2])[M.flatten() == 1].mean(0)


def get_std_masked(X, M):
    """
    Get the standard deviation for every feature in the batch of sequences X by
    considering only masked-in entries.
    :param X: Batch of sequences. shape = (time, sample, feature)
    :param M: Mask for the sequences. shape = (time, sample, 1)
    :returns: standard deviation of each feature. shape = (features, )
    """
    return X.reshape(-1, X.shape[2])[M.flatten() == 1].std(0)


def subtract_mean_masked(X, M, means):
    """
    Subtract the means from the masked-in entries of a batch of sequences X.

    :param X: Batch of sequences. shape = (time, sample, feature)
    :param M: Mask for the sequences. shape = (time, sample, 1)
    :param means: The means to subtract. shape = (features, )
    """
    for i in range(X.shape[2]):
        X[:, :, i][M[:, :, 0] == 1] -= means[i]


def divide_by_std_masked(X, M, stds):
    """
    Divide masked-in entries of X by the standard deviations stds.

    :param X: Batch of sequences. shape = (time, sample, feature)
    :param M: Mask for the sequences. shape = (time, sample, 1)
    :param stds: The standard deviations for every feature. shape = (features, )
    """
    for i in range(X.shape[2]):
        X[:, :, i][M[:, :, 0] == 1] /= stds[i]


def normalize_data(X_train, M_train, X_test, M_test):
    means = get_mean_masked(X_train, M_train)
    subtract_mean_masked(X_train, M_train, means)
    subtract_mean_masked(X_test, M_test, means)

    stds = get_std_masked(X_train, M_train)
    divide_by_std_masked(X_train, M_train, stds)
    divide_by_std_masked(X_test, M_test, stds)
    return means, stds


def shuffle_data(X, T, M):
    shuffling = range(X.shape[1])
    shuffle(shuffling)
    X_s = X[:, shuffling, :]
    T_s = T[:, shuffling, :]
    M_s = M[:, shuffling, :]
    return X_s, T_s, M_s, shuffling

def mid_pool_outputs(T, size=3):
    """
        Make pools of size=size (must be odd) such that targets of equal frames
        before and after are also available while training.
        """
    T_pooled = np.zeros((T.shape[0], T.shape[1], size*T.shape[2]))
    T = np.concatenate((np.zeros((size//2, T.shape[1], T.shape[2])), T, np.zeros((size//2, T.shape[1], T.shape[2]))), axis=0)
    for t in range(T.shape[0]-(size-1)):
        T_frame = T[t:t+size, :, :]
        T_pooled[t, :, :] = T_frame.reshape(1, size, T.shape[1], T.shape[2]).swapaxes(1, 2).reshape(1, T.shape[1], size*T.shape[2])
    return T_pooled



