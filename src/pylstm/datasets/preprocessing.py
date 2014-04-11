#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from pylstm.randomness import global_rnd
from pylstm.targets import Targets


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


def get_means(input_data, mask=None):
    """
    Get the mean values for every feature in the batch of sequences X by
    considering only masked-in entries.
    @param input_data: Batch of sequences. shape = (time, sample, feature)
    @param mask: Optional mask for the sequences. shape = (time, sample, 1)
    @return: mean value for each feature. shape = (features, )
    """
    if mask is not None:
        return input_data.reshape(-1, input_data.shape[2])[mask.flatten() == 1].mean(0)
    else:
        return input_data.mean((0, 1))


def get_stds(input_data, mask=None):
    """
    Get the standard deviation for every feature in the batch of sequences X by
    considering only masked-in entries.
    @param input_data: Batch of sequences. shape = (time, sample, feature)
    @param mask: Optional mask for the sequences. shape = (time, sample, 1)
    @return: standard deviation of each feature. shape = (features, )
    """
    if mask is not None:
        return input_data.reshape(-1, input_data.shape[2])[mask.flatten() == 1].std(0)
    else:
        return input_data.std((0, 1))


def subtract_means(input_data, means, mask=None):
    """
    Subtract the means from the masked-in entries of a batch of sequences X.
    This operation is performed in-place, i.e. the input_data will be modified.

    @param input_data: Batch of sequences. shape = (time, sample, feature)
    @param means: The means to subtract. shape = (features, )
    @param mask: Optional mask for the sequences. shape = (time, sample, 1)
    """
    if mask is not None:
        for i in range(input_data.shape[2]):
            input_data[:, :, i][mask[:, :, 0] == 1] -= means[i]
    else:
        input_data -= means


def divide_by_stds(input_data, stds, mask=None):
    """
    Divide masked-in entries of input_data by the stds.

    @param input_data: Batch of sequences. shape = (time, sample, feature)
    @param stds: The standard deviations for every feature. shape = (features, )
    @param mask: Optional mask for the sequences. shape = (time, sample, 1)
    """
    if mask is not None:
        for i in range(input_data.shape[2]):
            input_data[:, :, i][mask[:, :, 0] == 1] /= stds[i]
    else:
        input_data /= stds


def center_dataset(ds, channels_mask=None):
    """make the mean of each channel 0 over the training-set.
    If channels_mask is given, only consider the masked-in channels.
    """
    train = ds['training']
    if channels_mask is None:
        channels_mask = np.ones(train[0].shape[2], dtype=np.bool)
    assert channels_mask.shape == (train[0].shape[2], )
    assert channels_mask.dtype == np.bool

    input_data = train[0][:, :, channels_mask]
    means = get_means(input_data, train[1].mask)
    subtract_means(input_data, train[1].mask)

    if 'validation' in ds:
        subtract_means(ds['validation'][0][:, :, channels_mask],
                       means,
                       ds['validation'][1].mask)
    if 'test' in ds:
        subtract_means(ds['test'][0][:, :, channels_mask],
                       means,
                       ds['test'][1].mask)
    return means


def scale_std_of_dataset(ds, channels_mask=None):
    """make the std of each channel 1 over the training-set
    If channels_mask is given, only consider the masked-in channels.
    """
    train = ds['training']
    if channels_mask is None:
        channels_mask = np.ones(train[0].shape[2], dtype=np.bool)

    assert channels_mask.shape == (train[0].shape[2], )
    assert channels_mask.dtype == np.bool

    input_data = train[0][:, :, channels_mask]
    stds = get_stds(input_data, train[1].mask)
    divide_by_stds(input_data, stds, train[1].mask)
    if 'validation' in ds:
        divide_by_stds(ds['validation'][0][:, :, channels_mask],
                       stds,
                       ds['validation'][1].mask)
    if 'test' in ds:
        divide_by_stds(ds['test'][0][:, :, channels_mask],
                       stds,
                       ds['test'][1].mask)
    return stds


def normalize_dataset(ds, channels_mask=None):
    means = center_dataset(ds, channels_mask=channels_mask)
    stds = scale_std_of_dataset(ds, channels_mask=channels_mask)
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


def get_random_subset(ds, fraction=0.1):
    """
    Return a reduced dataset with only a fraction (default=0.1) of the data.
    """
    reduced_dataset = dict()
    for usage in ['training', 'validation', 'test']:
        input_data, targets = ds[usage]
        t, b, f = input_data.shape
        indices = np.arange(b)
        global_rnd['preprocessing'].shuffle(indices)
        reduced_size = round(fraction * b)
        reduced_indices = indices[:reduced_size]
        reduced_dataset[usage] = (input_data[:, reduced_indices, :],
                                  targets[reduced_indices])
    return reduced_dataset