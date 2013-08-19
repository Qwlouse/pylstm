#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import os
import numpy as np
from pylstm.datasets import MFCC

TIMIT_DIR = '.'

phonemes = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
            'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
            'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy',
            'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau',
            'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v',
            'w', 'y', 'z', 'zh']
silence_label = phonemes.index('h#')

reduced_phonemes = {
    'ae': 'aa',
    'ax': 'ah', 'ax-h': 'ah',
    'axr': 'er',
    'hv': 'hh',
    'ix': 'ih',
    'el': 'l',
    'em': 'm',
    'en': 'n', 'nx': 'n',
    'eng': 'ng',
    'zh': 'sh',
    'pcl': 'h#', 'tcl': 'h#', 'kcl': 'h#', 'bcl': 'h#', 'dcl': 'h#', 'gcl': 'h#', 'pau': 'h#', 'epi': 'h#',
    'ux': 'uw'
    # discard q
}


class TimitSample(object):
    @classmethod
    def create(cls, directory, name):
        f = os.path.join(directory, name.split('.')[0])
        f = f.split('/')[-4:]
        sample = cls(f[0], f[1], f[2][0], f[2][1:], f[3])
        return sample

    def __init__(self, usage, dialect, sex, speaker_id, sentence_id):
        self.usage = usage
        self.dialect = dialect
        self.sex = sex
        self.speaker_id = speaker_id
        self.sentence_id = sentence_id

    def _get_path(self, fileending, basedir):
        if not fileending.startswith('.'):
            fileending = '.' + fileending
        return os.path.join(basedir, self.usage, self.dialect, self.sex +
                            self.speaker_id, self.sentence_id + fileending)

    def get_sentence(self, basedir=TIMIT_DIR):
        filename = self._get_path('txt', basedir)
        with file(filename, 'r') as f:
            content = f.read()
            start, stop, sentence = content.split(' ', 2)
            return int(start), int(stop), sentence.strip()

    def get_words(self, basedir=TIMIT_DIR):
        filename = self._get_path('wrd', basedir)
        with file(filename, 'r') as f:
            content = f.readlines()
            wordlist = [c.strip().split(' ', 2) for c in content]
            return [(int(start), int(stop), word)
                    for start, stop, word in wordlist]

    def get_phonemes(self, basedir=TIMIT_DIR):
        filename = self._get_path('phn', basedir)
        with file(filename, 'r') as f:
            content = f.readlines()
            phoneme_list = [c.strip().split(' ', 2) for c in content]
            return [(int(start), int(stop), phoneme, phonemes.index(phoneme))
                    for start, stop, phoneme in phoneme_list]

    def get_audio_data(self, basedir=TIMIT_DIR):
        import scikits.audiolab as al
        filename = os.path.join(basedir, self.usage, self.dialect,
                                self.sex + self.speaker_id,
                                self.sentence_id + '.wav')
        f = al.Sndfile(filename, 'r')
        data = f.read_frames(f.nframes, dtype=np.float64)
        return data

    def get_labels(self, frame_size=400, frame_shift=160, basedir=TIMIT_DIR):
        phonemes = self.get_phonemes(basedir)
        p_extended = [silence_label] * (phonemes[0][0])
        for a in phonemes:
            p_extended += [a[3]] * (int(a[1]) - int(a[0]))
        end = phonemes[-1][1]
        windows = zip(range(0, end - frame_size + 1, frame_shift),
                      range(frame_size, end + 1, frame_shift))
        labels = [np.bincount(p_extended[f[0]:f[1]]).argmax() for f in windows]
        return np.array(labels)

    def get_features(self, basedir=TIMIT_DIR):
        d = self.get_audio_data(basedir)
        mfcc = MFCC.extract(d)
        mfcc_d1 = np.gradient(mfcc)[0]
        mfcc_d2 = np.gradient(mfcc_d1)[0]
        features = np.hstack((mfcc, mfcc_d1, mfcc_d2))
        labels = self.get_labels(basedir=basedir)
        return features, labels

    def __unicode__(self):
        return '<TimitSample ' + '/'.join([self.usage, self.dialect,
                                           self.sex + self.speaker_id,
                                           self.sentence_id]) + \
               '>'

    def __repr__(self):
        return self.__unicode__()


def read_all_samples(timit_dir=TIMIT_DIR):
    samples = []
    for dirname, dirnames, filenames in os.walk(timit_dir):
        samples += [TimitSample.create(dirname, n)
                    for n in filenames if n.endswith('.wav')]
    return samples


def filter_samples(samples, usage=None, dialect=None, sex=None, speaker_id=None,
                   sentence_id=None):
    def match(s):
        return (usage is None or s.usage == usage) and \
               (dialect is None or s.dialect == dialect) and \
               (sex is None or s.sex == sex) and \
               (speaker_id is None or s.speaker_id == speaker_id) and \
               (sentence_id is None or s.sentence_id == sentence_id)
    return [s for s in samples if match(s)]


def get_features_and_labels_for(samples, timit_dir=TIMIT_DIR, normalize=True):
    ds_list = [s.get_features(timit_dir) for s in samples]
    if normalize:
        features_mean = np.zeros((1, ds_list[0][0].shape[1]))
        features_num  = 0
        # Calculate the mean
        for f, _ in ds_list:
            features_mean = features_mean + f.sum(0)
            features_num  = features_num  + f.shape[0]
        features_mean = features_mean/features_num
        
        # Subtract the mean
        for f, _ in ds_list:
            f[:] = f[:] - features_mean
        
        # Calculate the std. dev.
        features_stddev = np.zeros((1, ds_list[0][0].shape[1]))
        for f, _ in ds_list:
            features_stddev = features_stddev + (f**2).sum(0)
        features_stddev = np.sqrt(features_stddev/features_num)
        
        # Normalize by the std. dev.
        for f, _ in ds_list:
            f[:] = f[:] / features_stddev
        
    maxlen = max(f.shape[0] for f, l in ds_list)
    padded_features = []
    padded_labels = []
    masks = []
    for f, l in ds_list:
        pad_length_f = maxlen - f.shape[0]
        pad_length_l = maxlen - l.shape[0]

        mask = np.ones_like(l)
        padded_features.append(np.vstack((f, np.zeros((pad_length_f, f.shape[1])))))
        padded_labels.append(np.hstack((l, np.ones(pad_length_l) * silence_label)))
        masks.append(np.hstack((mask, np.zeros(pad_length_l))))

    features = np.dstack(padded_features).swapaxes(1, 2)
    labels = np.vstack(padded_labels).T.reshape(maxlen, -1, 1)
    masks = np.vstack(masks).T.reshape(maxlen, -1, 1)
    return features, labels, masks


if __name__ == '__main__':
    timit_dir = '/home/greff/Datasets/realtimit/timit'
    samples = read_all_samples(timit_dir)
    print('allread')
    import time
    start = time.time()
    # train = filter_samples(samples, usage='train')
    # X, T, M = get_features_and_labels_for(train, timit_dir=timit_dir)
    # print(time.time() - start)
    # np.save(str('timit_train_X.numpy'), X)
    # np.save(str('timit_train_T.numpy'), T)
    # np.save(str('timit_train_M.numpy'), M)

    test = filter_samples(samples, usage='test')
    X, T, M = get_features_and_labels_for(test[:24], timit_dir=timit_dir)
    print(time.time() - start)
    print(X.shape)
    # np.save(str('timit_test_X.numpy'), X)
    # np.save(str('timit_test_T.numpy'), T)
    # np.save(str('timit_test_M.numpy'), M)

