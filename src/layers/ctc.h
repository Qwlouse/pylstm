#pragma once

#include <vector>
#include <map>

#include "matrix/matrix.h"


void ctc_alphas(Matrix Y_log, std::vector<int> T, Matrix alpha);
void ctc_betas(Matrix Y_log, std::vector<int> T, Matrix beta);


/**
 * Calculate the CTC error function and corresponding deltas given the network
 * outputs Y [matrix with (labels, 1, time)] and the targets T [vector of target
 * labels]. It will write the deltas [matrix (labels, 1, time)], and return
 * the error.
 */
double ctc(Matrix Y, std::vector<int> T, Matrix deltas);


/**
 * Decode the output of a softmax layer trained with CTC using a language model.
 * The so called CTC Token Passing Algorithm(Graves06, Graves08) is
 * based on the Token Passing algorithm for HMMs and DTW (Young89).
 * Inputs:
 *   - dict: list of lists of labels
 *   - onegrams: list of a-priori word log-probabilities
 *   - bigrams: map of wordpairs to log-probabilities
 *   - ln_y: Matrix of log network outputs
 * Returns:
 *   The most probable word-sequence
 */
std::vector<int> ctc_token_passing_decoding(
    std::vector<std::vector<int>> dict,
    std::vector<double> onegrams,
    std::map<std::pair<int, int>, double> bigrams,
    Matrix ln_y);