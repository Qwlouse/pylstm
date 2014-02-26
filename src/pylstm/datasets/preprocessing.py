#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from pylstm.randomness import global_rnd
from pylstm.targets import create_targets_object, Targets


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
    if len(A.shape) == 3:
        assert A.shape[2] == 1
        A = A[:, :, 0]
    if alphabet is None:
        alphabet = list(np.unique(A))
    else:
        alphabet = list(alphabet)
    result = np.zeros((A.shape[0], A.shape[1], len(alphabet)))
    for i, a in enumerate(alphabet):
        result[A == a, i] = 1
    return result


def get_mean_masked(X, M):
    """
    Get the mean values for every feature in the batch of sequences X by
    considering only masked-in entries.
    @param X: Batch of sequences. shape = (time, sample, feature)
    @param M: Mask for the sequences. shape = (time, sample, 1)
    @return: mean value for each feature. shape = (features, )
    """
    return X.reshape(-1, X.shape[2])[M.flatten() == 1].mean(0)


def get_std_masked(X, M):
    """
    Get the standard deviation for every feature in the batch of sequences X by
    considering only masked-in entries.
    @param X: Batch of sequences. shape = (time, sample, feature)
    @param M: Mask for the sequences. shape = (time, sample, 1)
    @return: standard deviation of each feature. shape = (features, )
    """
    return X.reshape(-1, X.shape[2])[M.flatten() == 1].std(0)


def subtract_mean_masked(X, M, means):
    """
    Subtract the means from the masked-in entries of a batch of sequences X.

    @param X: Batch of sequences. shape = (time, sample, feature)
    @param M: Mask for the sequences. shape = (time, sample, 1)
    @param means: The means to subtract. shape = (features, )
    """
    for i in range(X.shape[2]):
        X[:, :, i][M[:, :, 0] == 1] -= means[i]


def divide_by_std_masked(X, M, stds):
    """
    Divide masked-in entries of X by the standard deviations stds.

    @param X: Batch of sequences. shape = (time, sample, feature)
    @param M: Mask for the sequences. shape = (time, sample, 1)
    @param stds: The standard deviations for every feature. shape = (features, )
    """
    for i in range(X.shape[2]):
        X[:, :, i][M[:, :, 0] == 1] /= stds[i]


def normalize_data(X_train, M_train, X_test, M_test):
    """
    why does this shit not work?


    @param X_train: the training input data
    @type X_train ndarray
    @param M_train:
    @type M_train ndarray
    @param X_test:
    @param M_test:

    @rtype : (ndarray, ndarray)
    """
    means = get_mean_masked(X_train, M_train)
    subtract_mean_masked(X_train, M_train, means)
    subtract_mean_masked(X_test, M_test, means)

    stds = get_std_masked(X_train, M_train)
    divide_by_std_masked(X_train, M_train, stds)
    divide_by_std_masked(X_test, M_test, stds)
    return means, stds


def shuffle_data(input_data, targets, seed=None):
    """
    Shuffles the samples of the data.

    @param input_data: Batch of sequences
    @type input_data: ndarray
    @type targets: pylstm.targets.Targets
    @param targets: Targets for the sequences
    @type seed: int | None

    @return: A tuple (input_data_shuffled, targets_shuffled, indices), where
             input_data_shuffled and targets_shuffled are the shuffled
             input_data and targets respectively, and indices is the list
             of shuffling indices.
    """
    assert isinstance(targets, Targets)
    indices = np.arange(input_data.shape[1])
    global_rnd['preprocessing'].get_new_random_state(seed).shuffle(indices)
    input_data_shuffled = input_data[:, indices, :]
    targets_shuffled = targets[indices]

    return input_data_shuffled, targets_shuffled, indices


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


def mask_features(ds, feature_mask):
    """
    Can be used to remove some features from a dataset.
    @param ds: dataset dictionary
    @param feature_mask: binary mask with shape = (# features, )
    @return: new ds dictionary
    """
    masked_ds = {}
    for usage in ds:
        if ds[usage] is None:
            continue
        input_data, targets = ds[usage]
        input_data = input_data[:, :, feature_mask == 1]
        masked_ds[usage] = input_data, targets
    return masked_ds