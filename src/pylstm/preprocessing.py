#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from random import shuffle


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



