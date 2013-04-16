#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from scipy.optimize import approx_fprime
import numpy as np
from pylstm.wrapper import Buffer


def ensure_np_array(a):
    if isinstance(a, Buffer):
        return a.as_array()
    else:
        return np.array(a)


class ErrorFunction(object):
    def evaluate(self, Y, T):
        pass

    def deriv(self, Y, T):
        pass


class MeanSquaredError(ErrorFunction):
    def evaluate(self, Y, T):
        Y = ensure_np_array(Y)
        return 0.5 * np.sum((Y - T) ** 2)

    def deriv(self, Y, T):
        Y = ensure_np_array(Y)
        return Y - T


class CrossEntropyError(object):
    def __call__(self, Y, T):
        Y = ensure_np_array(Y)
        Y[Y < 1e-6] = 1e-6
        cee = T * np.log(Y)
        return - np.sum(cee)

    def evaluate(self, Y, T):
        return self(Y, T)

    def deriv(self, Y, T):
        Y = ensure_np_array(Y)
        return - T / Y


class Accuracy(object):
    def __call__(self, Y, T):
        t, b, f = Y.shape
        Y = ensure_np_array(Y)
        winner_Y = np.argmax(Y, 2)
        winner_T = np.argmax(T, 2)
        return np.sum(winner_Y == winner_T) / (t * b)

    def evaluate(self, Y, T):
        return self(Y, T)


class CTC(object):
    def __call__(self, Y, T):
        # Y are network outputs with one output for each label plus the blank
        # blank label is index 0
        # T is the label sequence It does not have to have the same length
        # sanity checks:
        N, batch_size, label_count = Y.shape
        S, b, tmp = T.shape
        assert batch_size == b
        assert tmp == 1
        required_time = S
        previous_labels = -np.ones((b,))
        T = T[:, :, 0]
        for s in range(S):
            required_time += T[s, :] == previous_labels
            previous_labels = T[s, :]
        assert np.all(required_time <= N)
        labels = np.unique(T)
        assert len(labels) + 1 == label_count
        Z = 2 * S + 1
        # calculate forward variables alpha
        ## set up the dynamic programming matrix
        alpha = np.zeros((N, b, Z))
        alpha[0, :, 0] = Y[0, :, 0]
        alpha[0, :, 1] = Y[0, range(b), T[0, :]]
        for t in range(1, N):
            start = max(-1, 2 * (S - N + t) + 1)
            for s in range(start + 1, Z, 2):  # loop the even ones (blanks)
                alpha[t, :, s] += alpha[t - 1, :, s]
                if s > 0:
                    alpha[t, :, s] += alpha[t - 1, :, s - 1]
                alpha[t, :, s] *= Y[t, :, 0]
            previous_labels = -np.ones((b,))
            for s in range(max(1, start), Z, 2):  # loop the odd ones (labels)
                alpha[t, :, s] += alpha[t - 1, :, s]
                alpha[t, :, s] += alpha[t - 1, :, s - 1]
                labels = T[s // 2, :]
                if s > 1:
                    alpha[t, :, s] += alpha[t - 1, :, s - 2] * (labels != previous_labels)
                alpha[t, :, s] *= Y[t, :, labels][:, 0]
                previous_labels = labels

        beta = np.zeros((N, b, Z))
        beta[N - 1, :, 2 * S - 1] = 1
        beta[N - 1, :, 2 * S] = 1
        for t in range(N - 1, 0, -1):
            stop = min(Z, 2 * t)
            for s in range(0, stop, 2):  # loop the even ones (blanks)
                beta[t - 1, :, s] += beta[t, :, s] * Y[t, :, 0]
                if s < Z - 1:
                    labels = T[(s + 1) // 2, :]
                    beta[t - 1, :, s] += beta[t, :, s + 1] * Y[t, :, labels][:, 0]
            previous_labels = -np.ones((b,))
            for s in range(1, stop, 2):  # loop the odd ones (labels)
                labels = T[s // 2, :]
                beta[t - 1, :, s] += beta[t, :, s] * Y[t, :, labels][:, 0]
                beta[t - 1, :, s] += beta[t, :, s + 1] * Y[t, :, 0]
                if s < Z - 2:
                    labels = T[(s + 2) // 2, :]
                    beta[t - 1, :, s] += beta[t, :, s + 2] * Y[t, :, labels][:, 0] * (labels != previous_labels)
                previous_labels = labels

        ppix = alpha * beta
        pzx = ppix.sum(2)
        deltas = np.zeros((N, b, label_count))

        deltas[:, :, 0] = ppix[:, :, ::2].sum(2)
        for s in range(1, Z, 2):
            deltas[:, :, T[s // 2, :]] += ppix[:, :, s:s + 1]
        for l in range(label_count):
            deltas[:, :, l] /= - Y[:, :, l] * pzx

        return alpha, beta, deltas

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
    print("loss =", -np.log((a * b).T.sum(0).mean()))
    print("deltas\n", d.T)

    # finite differences testing
    def f(X):
        a, b, d = c(X.reshape(4, 1, 3), T)
        return -np.log((a * b).T.sum(0).mean())

    delta_approx = approx_fprime(Y.copy().flatten(), f, 1e-5)
    print("delta_approx\n", delta_approx.reshape(4, 3).T)