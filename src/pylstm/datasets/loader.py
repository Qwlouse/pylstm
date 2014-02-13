#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals

import os
import numpy as np
import cPickle
from pylstm.targets import SequencewiseTargets


def get_files_containing(file_list, search_string, ignore_case=False):
    if ignore_case:
        return [c for c in file_list if os.path.basename(c).lower().find(search_string.lower()) != -1]
    else:
        return [c for c in file_list if os.path.basename(c).find(search_string) != -1]


def load_data(files):
    """
    Picks the shortest filename among the options and loads it if it has a
    .npy or .pickle ending.
    """
    filename = None
    if isinstance(files, list):
        for f in sorted(files):
            filename = f if filename is None or len(f) < len(filename) else filename
    else:
        filename = files
    print('loading "%s"' % filename)
    if filename.endswith('.npy'):
        return np.load(filename)
    if filename.endswith('.pickle'):
        with open(filename, 'r') as f:
            return cPickle.load(f)
    return None


def read_data(candidates, targets='T'):
    X = load_data(get_files_containing(candidates, 'X'))
    T = load_data(get_files_containing(candidates, targets))
    M_candidates = get_files_containing(candidates, 'M')
    if M_candidates:
        M = load_data(M_candidates)
    else:
        M = np.ones((T.shape[0], T.shape[1], 1))
    return X, T, M


def load_dataset(dataset_path, subset='', targets='T'):
    """
    Tries to load a dataset from the given path. It will look for the filenames
    to populate a dictionary with 'train', 'val' and 'test' sets. Each set
    consisting of an 'X', 'T' and possibly 'M' matrix.
    You can restrict the matching filesnames by passing a subset string.
    In case of ambiguity it will load the file with the shortest filename.
    """
    candidates = [os.path.join(dataset_path, p)
                  for p in os.listdir(dataset_path)]
    if subset:
        candidates = get_files_containing(candidates, subset)

    train_files = get_files_containing(candidates, 'train', True)
    val_files = get_files_containing(candidates, 'val', True)
    test_files = get_files_containing(candidates, 'test', True)

    ds = dict()
    ds['train'] = read_data(train_files, targets)
    ds['test'] = read_data(test_files, targets) if test_files else None
    ds['val'] = read_data(val_files, targets) if val_files else None
    return ds


def transform_ds_to_nsp(ds):
    """
    Takes a dataset dictionary like the one returned from load_dataset
    and transforms it into a next-step-prediction task.
    """
    ds_nsp = {}
    for use in ds:
        if ds[use] is None:
            continue
        ds_nsp[use] = (ds[use][0][:-1, :, :],
                       ds[use][0][1:, :, :],
                       ds[use][2][:-1, :, :])
    return ds_nsp


def transform_ds_to_seq_classification(ds):
    """
    Takes a dataset dictionary like the one returned from load_dataset
    and transforms it into a sequence classification task.
    """
    classes = list(np.lib.arraysetops.unique(ds['train'][1]))
    ds_seq_class = {}
    for use in ds:
        if ds[use] is None:
            continue
        T = np.array([classes.index(t) for t in ds[use][1].flatten()])
        ds_seq_class[use] = (ds[use][0],
                             SequencewiseTargets(T, binarize_to=len(classes)),
                             ds[use][2])

    return ds_seq_class, len(classes)


def mask_features(ds, feature_mask):
    """
    Can be used to remove some features from a dataset.
    :param ds: dataset dictionary
    :param feature_mask: binary mask with shape = (# features, )
    :return: new ds dictionary
    """
    masked_ds = {}
    for usage in ds:
        if ds[usage] is None:
            continue
        X, T, M = ds[usage]
        X = X[:, :, feature_mask == 1]
        masked_ds[usage] = X, T, M
    return masked_ds