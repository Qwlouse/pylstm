#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals

import os
import numpy as np
import cPickle
from pylstm.targets import SequencewiseTargets, create_targets_object, \
    FramewiseTargets, LabelingTargets


def get_files_containing(file_list, search_string, ignore_case=False):
    if ignore_case:
        return [c for c in file_list if os.path.basename(c).lower().find(search_string.lower()) != -1]
    else:
        return [c for c in file_list if os.path.basename(c).find(search_string) != -1]


def load_data(files):
    """
    Picks the shortest filename among files and loads it if it has a
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
    elif filename.endswith('.pickle'):
        with open(filename, 'r') as f:
            return cPickle.load(f)
    return None


def read_data(candidates, targets='T'):
    input_data = load_data(get_files_containing(candidates, 'X'))
    targets_data = load_data(get_files_containing(candidates, targets))
    mask_candidates = get_files_containing(candidates, 'M')
    if mask_candidates:
        mask = load_data(mask_candidates)
    else:
        mask = None
    targets = create_targets_object(targets_data, mask)
    return input_data, targets


def load_dataset_collection(dataset_path, subset='', targets='T'):
    """
    Tries to load a dataset from the given path. It will look for the filenames
    to populate a dictionary with 'train', 'val' and 'test' sets. Each set
    consisting of an 'X', 'T' and possibly 'M' matrix.
    You can restrict the matching filesnames by passing a subset string.
    In case of ambiguity it will load the file with the shortest filename.
    """
    candidates = [os.path.join(dataset_path, p)
                  for p in os.listdir(dataset_path)
                  if os.path.isfile(os.path.join(dataset_path, p))]
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
    for use in ['training', 'validation', 'test']:
        if use not in ds or ds[use] is None:
            continue
        nsp_targets = create_targets_object(ds[use][0][1:])
        ds_nsp[use] = (ds[use][0][:-1, :, :],
                       nsp_targets)
    return ds_nsp


class Dataset(dict):
    def __init__(self):
        super(Dataset, self).__init__()
        self['description'] = ''

    @property
    def description(self):
        return self['description']

    @property
    def training(self):
        return self['training']

    @property
    def validation(self):
        return self['validation']

    @property
    def test(self):
        return self['test']


def load_dataset(filename, variant=''):
    import h5py
    ds = Dataset()
    with h5py.File(filename, "r") as f:
        if variant:
            assert variant in f
            v = f[variant]
        elif 'default' in f:
            v = f['default']
        else:
            v = f

        # get description
        if 'description' in f.attrs:
            ds['description'] = f.attrs['description']
        if v != f and 'description' in v.attrs:
            ds['description'] += "\nVariant: %s\n=============\n\n" % variant +\
                                 v.attrs['description']

        for usage in ['training', 'validation', 'test']:
            if usage not in v:
                continue
            grp = v[usage]
            # read input_data from group
            assert 'input_data' in grp, "Did not find input_data for " + usage
            input_data = grp['input_data'][:]

            # read targets from group
            assert 'targets' in grp, "Did not find targets for " + usage
            targets_ds = grp['targets']
            if 'targets_type' in targets_ds.attrs:
                targets_type = targets_ds.attrs['targets_type']
            else:
                raise RuntimeError('No targets_type attribute found!')

            if 'binarize_to' in targets_ds.attrs:
                binarize_to = targets_ds.attrs['binarize_to']
                if binarize_to <= 0:
                    binarize_to = None
            else:
                binarize_to = None
            # read mask
            mask = grp['mask'][:] if 'mask' in grp else None

            # create targets object
            if targets_type == 'F':
                targets = FramewiseTargets(targets_ds[:], mask, binarize_to)
            elif targets_type == 'L':
                # convert targets_data to list of lists
                targets_list = cPickle.loads(targets_ds.value.tostring())
                targets = LabelingTargets(targets_list, mask, binarize_to)
            elif targets_type == 'S':
                targets = SequencewiseTargets(targets_ds[:], mask, binarize_to)
            else:
                raise ValueError('Unsupported targets_type "%s"' % targets_type)

            ds[usage] = input_data, targets
    return ds


def get_chunksize(data):
    t, b, f = data.shape
    size_of_sequence = t*f*8
    seqs_per_chunk = min((10240 // size_of_sequence) + 1, b)
    chunksize = (t, seqs_per_chunk, f)
    return chunksize


def save_dataset_as_hdf5(dataset, filename=None, variant=None):
    """
    Method to write simple datasets to an HDF5 file.

    :param dataset: The dataset to be stored as a dictionary of tuples.
        Each entry is one usage and contains (input_data, targets)
    :type dataset: dict[unicode, (numpy.ndarray, pylstm.targets.Targets)]

    :param filename: Filename/path of the file that should be written.
        Will overwrite if it already exists. Can be None if variant is given.
    :type filename: unicode

    :param variant: hdf5 group object the dataset will be saved to instead of
        writing it to a new file. Either this or filename has to be set.

    :rtype: None
    """
    hdffile = None
    if variant is None:
        assert filename is not None
        import h5py
        hdffile = h5py.File(filename, "w")
        variant = hdffile

    if 'description' in dataset:
        variant.attrs['description'] = dataset['description']
    for usage in ['training', 'validation', 'test']:
        if usage not in dataset:
            continue
        input_data, targets = dataset[usage]
        grp = variant.create_group(usage)

        grp.create_dataset('input_data', data=input_data,
                           chunks=get_chunksize(input_data),
                           compression="gzip")

        if targets.is_labeling():
            targets_encoded = np.void(cPickle.dumps(targets.data))
            targets_ds = grp.create_dataset('targets',
                                            data=targets_encoded,
                                            dtype=targets_encoded.dtype)
        else:
            targets_ds = grp.create_dataset(
                'targets',
                data=targets.data,
                chunks=get_chunksize(targets.data),
                compression="gzip"
            )
        targets_ds.attrs.create('targets_type', str(targets.targets_type[0]))
        targets_ds.attrs.create('binarize_to', targets.binarize_to or 0)
        if targets.mask is not None:
            grp.create_dataset('mask', data=targets.mask, dtype='u1')

    if hdffile is not None:
        hdffile.close()


def save_dataset_varianst_as_hdf5(variants, filename):
    import h5py
    hdffile = h5py.File(filename, "w")
    for var, ds in variants.items():
        variant = hdffile.create_group(var)
        save_dataset_as_hdf5(ds, variant=variant)
    hdffile.close()
