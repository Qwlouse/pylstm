#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from scipy.optimize import approx_fprime
import numpy as np


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

    return np.exp(alpha)


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
    return np.exp(beta)



def CTC(Y, T, M=None):
    # TODO remove multibatch support and move it to a loop
    import warnings
    with warnings.catch_warnings():
        # This removes all the warnings about -inf in logaddexp
        # those values are necessary and the results are correct
        warnings.simplefilter("ignore")

        # Y are network outputs with one output for each label plus the blank
        # blank label is index 0
        # T is the label sequence It does not have to have the same length
        # sanity checks:
        N, label_count = Y.shape
        S = len(T)
        required_time = S
        previous_label = -1
        #T = T[:, :, 0]
        for s in range(S):
            required_time += T[s] == previous_label
            previous_label = T[s]
        assert np.all(required_time <= N)
        labels = np.unique(T)
        assert len(labels) + 1 <= label_count
        Z = 2 * S + 1
        #### Convert Y to log scale:
        Y_log = np.log(Y)
        # calculate forward variables alpha
        ## set up the dynamic programming matrix

        alpha = ctc_calculate_alphas(Y_log, T)
        beta = ctc_calculate_betas(Y_log, T)

        ppix = alpha * beta
        pzx = ppix.sum(1)
        deltas = np.zeros((N, label_count))

        deltas[:, 0] = ppix[:, ::2].sum(1)
        for s in range(1, Z, 2):
            deltas[:,  T[s // 2]] += ppix[:, s]
        for l in range(label_count):
            deltas[:, l] /= - Y[:, l] * pzx

        error = -(np.log(ppix.sum(1))).mean()

        return error, deltas


if __name__ == "__main__":
    Y = np.array([[.1, .7, .2], [.8, .1, .1], [.3, .3, .4], [.7, .1, .2]]).reshape(4, 1, 3)
    T = np.array([1, 2]).reshape(-1, 1, 1)
    c = CTC()
    a, b, d = c(Y, T)
    a_expected = np.array([[.1, .08, 0, 0], [.7, .08, .048, 0], [0, .56, .192, 0], [0, .07, .284, .1048], [0, 0, .021, .2135]])
    b_expected = np.array([[.096, .06, 0, 0], [.441, .48, .2, 0], [0, .42, .2, 0], [0, .57, .9, 1], [0, 0, .7, 1]])
    print("alphas\n", a.T)
    print("betas\n", b.T)
    print("p(z|x) =", (a * b).T.sum(0) ) # should all be equal
    print("loss =", -(np.log((a * b).sum(2))).sum(1).mean())
    print("deltas\n", d.T)

    # finite differences testing
    def f(X):
        a, b, d = c(X.reshape(4, 1, 3), T)
        return -np.log((a * b).T.sum(0).mean())

    delta_approx = approx_fprime(Y.copy().flatten(), f, 1e-5)
    print("delta_approx\n", delta_approx.reshape(4, 3).T)