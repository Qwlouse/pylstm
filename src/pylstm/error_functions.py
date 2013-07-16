#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from trainer import Online


def MeanSquaredError(Y, T, M=None):
    diff = Y - T
    norm = Y.shape[0] * Y.shape[1]
    if M is not None:
        diff *= M
        norm = M.sum()
    error = 0.5 * np.sum(diff ** 2) / norm
    deltas = diff / norm
    return error, deltas


def CrossEntropyError(Y, T, M=None):
    Y = Y.copy()  # do not modify original Y
    Y[Y < 1e-6] = 1e-6
    cee = T * np.log(Y) + (1 - T) * np.log(1 - Y)
    ceed = (T - Y) / (Y * (Y - 1))
    norm = Y.shape[0] * Y.shape[1]
    if M is not None:
        cee *= M
        ceed *= M
        norm = M.sum()
    error = - np.sum(cee) / norm
    deltas = ceed / norm
    return error, deltas


def MultiClassCrossEntropyError(Y, T, M=None):
    Y = Y.copy()  # do not modify original Y
    Y[Y < 1e-6] = 1e-6
    cee = T * np.log(Y)
    quot = T / Y
    norm = Y.shape[0] * Y.shape[1]
    if M is not None:
        cee *= M
        quot *= M
        norm = M.sum()
    error = - np.sum(cee) / norm
    deltas = - quot / norm
    return error, deltas


neg_inf = float('-inf')


def ctc_calculate_alphas(Y_log, T):
    """
    Y_log: log of outputs shape=(time, labels)
    T: target sequence shape=(length, )
    """
    N = Y_log.shape[0]
    S = len(T)
    Z = 2 * S + 1

    alpha = np.zeros((N, Z))
    alpha[:] = neg_inf
    alpha[0, 0] = Y_log[0, 0]
    alpha[0, 1] = Y_log[0, T[0]]
    for t in range(1, N):
        start = max(-1, 2 * (S - N + t) + 1)
        for s in range(start + 1, Z, 2):  # loop the even ones (blanks)
            alpha[t, s] = np.logaddexp(alpha[t, s], alpha[t - 1, s])
            if s > 0:
                alpha[t, s] = np.logaddexp(alpha[t, s], alpha[t - 1, s - 1])

            alpha[t, s] += Y_log[t, 0]
        previous_label = -1
        if start > 0:
            previous_label = T[start // 2 - 1]
        for s in range(max(1, start), Z, 2):  # loop the odd ones (labels)
            alpha[t, s] = np.logaddexp(alpha[t, s], alpha[t - 1, s])
            alpha[t, s] = np.logaddexp(alpha[t, s], alpha[t - 1, s - 1])
            label = T[s // 2]
            if s > 1:
                alpha[t, s] = np.logaddexp(alpha[t, s], alpha[t - 1, s - 2] + np.log(label != previous_label))
            alpha[t, s] += Y_log[t, label]
            previous_label = label

    return alpha


def ctc_calculate_betas(Y_log, T):
    N = Y_log.shape[0]
    Z = 2 * len(T) + 1

    beta = np.zeros((N, Z))
    beta[:] = neg_inf
    beta[N - 1, Z - 2] = 0.0
    beta[N - 1, Z - 1] = 0.0
    for t in range(N - 1, 0, -1):
        stop = min(Z, 2 * t)
        for s in range(0, stop, 2):  # loop the even ones (blanks)
            beta[t - 1, s] = np.logaddexp(beta[t - 1, s], beta[t, s] + Y_log[t, 0])
            if s < Z - 1:
                label = T[(s + 1) // 2]
                beta[t - 1, s] = np.logaddexp(beta[t - 1, s], beta[t, s + 1] + Y_log[t, label])
        for s in range(1, stop, 2):  # loop the odd ones (labels)
            label = T[s // 2]
            beta[t - 1, s] = np.logaddexp(beta[t - 1, s], beta[t, s] + Y_log[t, label])
            beta[t - 1, s] = np.logaddexp(beta[t - 1, s], beta[t, s + 1] + Y_log[t, 0])
            if s < Z - 2:
                previous_label = label
                label = T[(s + 2) // 2]
                if label != previous_label:
                    beta[t - 1, s] = np.logaddexp(beta[t - 1, s], beta[t, s + 2] + Y_log[t, label])
    return beta

def CTC(Y, T, M=None):
    import warnings
    with warnings.catch_warnings():
        # This removes all the warnings about -inf in logaddexp
        # those values are necessary and the results are correct
        warnings.simplefilter("ignore")

        # Y are network outputs with one output for each label plus the blank
        # blank label is index 0
        # T is the label sequence It does not have to have the same length
        N, batch_size, label_count = Y.shape

        #### Convert Y to log scale:
        Y_log = np.log(Y)
        # calculate forward variables alpha
        ## set up the dynamic programming matrix
        deltas = np.zeros((N, batch_size, label_count))
        deltas[:] = neg_inf
        errors = []
        for b, (y, t, m) in enumerate(Online(Y_log, T, M)):
            t = t[0]
            y = y.reshape(-1, label_count)
            # check required time is met
            S = len(t)
            required_time = S
            previous_label = -1
            for s in range(S):
                required_time += t[s] == previous_label
                previous_label = t[s]
            assert required_time <= y.shape[0]

            alpha = ctc_calculate_alphas(y, t)
            beta = ctc_calculate_betas(y, t)

            ppix = alpha + beta
            pzx = np.logaddexp.reduce(ppix, axis=1)

            deltas[:, b, 0] = np.logaddexp.reduce(ppix[:, ::2], axis=1)
            for s in range(1, 2 * S + 1, 2):
                deltas[:, b, t[s // 2]] = np.logaddexp(deltas[:, b, t[s // 2]], ppix[:, s])
            for l in range(label_count):
                deltas[:, b, l] -= y[:, l] + pzx

            errors.append(-pzx.mean())

        return np.mean(errors), -np.exp(deltas) / batch_size

