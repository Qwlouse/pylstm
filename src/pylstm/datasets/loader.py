#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals

import os
import numpy as np
import cPickle


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


def read_data(candidates):
    X = load_data(get_files_containing(candidates, 'X'))
    T = load_data(get_files_containing(candidates, 'T'))
    M_candidates = get_files_containing(candidates, 'M')
    if M_candidates:
        M = load_data(M_candidates)
    else:
        M = np.ones(T.shape[0], T.shape[1], 1)
    return X, T, M


def load_dataset(base_path, restriction=''):
    candidates = [os.path.join(base_path, p) for p in os.listdir(base_path)]
    if restriction:
        candidates = get_files_containing(candidates, restriction)

    train_files = get_files_containing(candidates, 'train', True)
    val_files = get_files_containing(candidates, 'val', True)
    test_files = get_files_containing(candidates, 'test', True)

    ds = dict()
    ds['train'] = read_data(train_files)
    ds['test'] = read_data(test_files) if test_files else None
    ds['val'] = read_data(val_files) if val_files else None
    return ds
