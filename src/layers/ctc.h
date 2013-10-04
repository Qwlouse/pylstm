#pragma once

#include <vector>

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