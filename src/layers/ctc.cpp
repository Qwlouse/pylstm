
#include <limits>
#include <algorithm>
#include <cmath>

#include "ctc.h"

#include "core/Assert.h"
#include "matrix/matrix.h"
#include "matrix/matrix_operation.h"

double ninf = - std::numeric_limits<double>::infinity();

/*
exp(0) = 1
exp(ninf) = 0
exp(inf) = inf
log(1) = 0
log(0) = -inf
log(-1) = nan
*/
void logaddexp(double& a, const double& b) {
    // computes :     a = log(exp(a) + exp(b))
    // equivalent to: a = a + log(1.0 + exp(b-a))
    // equivalent to: a = b + log(1.0 + exp(a-b))

    if (a >= b) {
        if (b == ninf) return;
        a = a + log(1.0 + exp(b - a));
    } else {
        if (a == ninf) {
             a = b;
             return;
        }
        a = b + log(1.0 + exp(a - b));
    }
}


// alpha.get(Z, 0, N)
// Y_log.get(F, 0, N)
//     T[S]
void ctc_alphas(Matrix Y_log, std::vector<int> T, Matrix alpha) {
    ASSERT(Y_log.n_columns == 1);
    int N = static_cast<int>(Y_log.n_slices);
    int S = static_cast<int>(T.size());
    int Z = static_cast<int>(alpha.n_rows);
    int label_count = static_cast<int>(Y_log.n_rows);
    ASSERT(alpha.n_rows == Z);
    ASSERT(alpha.n_columns == 1);
    ASSERT(alpha.n_slices = N);

    alpha.set_all_elements_to(ninf);
    alpha.get(0, 0, 0) = Y_log.get(0, 0, 0);
    alpha.get(1, 0, 0) = Y_log.get(T[0]+1, 0, 0);

    for (int t = 1; t < N; ++t) {
        int start = std::max(-1, 2 * (S - N + t) + 1);
        for (int s = start + 1; s < Z; s += 2) {    // loop the even ones
            double& current_alpha = alpha.get(s, 0, t);
            logaddexp(current_alpha, alpha.get(s, 0, t-1));
            if (s > 0) {
                logaddexp(current_alpha, alpha.get(s-1, 0, t-1));
            }
            current_alpha += Y_log.get(0, 0, t);
        }
        int previous_label = -1;
        if (start > 0) {
            previous_label = static_cast<int>(T[start / 2 - 1] + 1);
            ASSERT(previous_label < label_count);
        }
        for (int s = std::max(1, start); s < Z; s += 2) { // loop the odd ones (labels)
            double& current_alpha = alpha.get(s, 0, t);
            logaddexp(current_alpha, alpha.get(s, 0, t-1));
            logaddexp(current_alpha, alpha.get(s-1, 0, t-1));
            int label = static_cast<int>(T[s / 2] + 1);
            ASSERT(label < label_count);
            if ((s > 1) && (label != previous_label)) {
                logaddexp(current_alpha, alpha.get(s-2, 0, t-1));
            }
            current_alpha += Y_log.get(label, 0, t);
            previous_label = label;
        }
    }
}

//  beta.get(Z, 0, N)
// Y_log.get(F, 0, N)
//     T[S]
void ctc_betas(Matrix Y_log, std::vector<int> T, Matrix beta) {
    ASSERT(Y_log.n_columns == 1);
    int N = static_cast<int>(Y_log.n_slices);
    //int S = static_cast<int>(T.size());
    int Z = static_cast<int>(beta.n_rows);
    ASSERT(beta.n_rows == Z);
    ASSERT(beta.n_columns == 1);
    ASSERT(beta.n_slices = N);

    beta.set_all_elements_to(ninf);
    beta.get(Z-2, 0, N-1) = 0.0;
    beta.get(Z-1, 0, N-1) = 0.0;

    for (int t = N-1; t > 0; --t) {
        int stop = std::min(Z, 2 * t);
        for (int s = 0; s < stop; s += 2) {    // loop the even ones
            double& current_beta = beta.get(s, 0, t - 1);
            logaddexp(current_beta, beta.get(s, 0, t) + Y_log.get(0, 0, t));
            if (s < Z - 1) {
                int label = T[(s + 1) / 2] + 1;
                logaddexp(current_beta, beta.get(s+1, 0, t) + Y_log.get(label, 0, t));
            }
        }
        for (int s = 1; s < stop; s += 2) { // loop the odd ones (labels)
            double& current_beta = beta.get(s, 0, t - 1);
            int label = T[s / 2] + 1;
            logaddexp(current_beta, beta.get(s, 0, t) + Y_log.get(label, 0, t));
            logaddexp(current_beta, beta.get(s+1, 0, t) + Y_log.get(0, 0, t));

            if (s < Z - 2) {
                int next_label = T[(s + 2) / 2] + 1;
                if (label != next_label) {
                    logaddexp(current_beta, beta.get(s+2, 0, t) + Y_log.get(next_label, 0, t));
                }
            }
        }
    }
}

// deltas.get(Y, 0, N)
// alpha.get(Z, 0, N)
//  beta.get(Z, 0, N)
// Y_log.get(F, 0, N)
//     T[S]
double ctc(Matrix Y, std::vector<int> T, Matrix deltas) {
    int N = static_cast<int>(Y.n_slices);
    ASSERT(Y.n_columns == 1); // No multibatch support so far
    int label_count = static_cast<int>(Y.n_rows);
    ASSERT(deltas.n_rows == label_count);
    ASSERT(deltas.n_columns == 1);
    ASSERT(deltas.n_slices == N);
    // convert all outputs to log scale
    Matrix Y_log(label_count, 1, N);
    apply(Y, Y_log, log);

    int S = static_cast<int>(T.size());

    // check that the required time is met
    int required_time = S;
    int previous_label = -1;
    for (int s = 0; s < S; ++s) {
        int label = T[s] + 1;
        required_time += label == previous_label;
        previous_label = label;
    }
    ASSERT(required_time <= N);

    // calculate alphas and betas (in log scale)
    int Z = 2 * S + 1;
    Matrix alpha(Z, 1, N);
    ctc_alphas(Y_log, T, alpha);
    Matrix beta(Z, 1, N);
    ctc_betas(Y_log, T, beta);

    // add them to get ppix = alpha + beta
    add_into_b(alpha, beta);
    // beta now holds ppix

    // logaddexp over features for every timestep to get the normalization term
    Matrix pzx(1, 1, N);
    pzx.set_all_elements_to(ninf);
    for (int t = 0; t < N; ++t) {
        for (int s = 0; s < Z; ++s) {
            logaddexp(pzx[t], beta.get(s, 0, t));
        }
    }

    double error = 0.0;
    for (int t = 0; t < N; ++t) {
        // calculate deltas for the even ones (empty-label)
        for (int s = 0; s < Z; s += 2) {
            logaddexp(deltas.get(0, 0, t), beta.get(s, 0, t));
        }
        // calculate deltas for the odd ones (labels)
        for (int s = 1; s < Z; s += 2) {
            int label = T[s / 2] + 1;
            logaddexp(deltas.get(label, 0, t), beta.get(s, 0, t));
        }
        // normalize all the labels
        for (int l = 0; l < label_count; ++l) {
            deltas.get(l, 0, t) -= Y_log.get(l, 0, t) + pzx[t];
        }
        // mean of -pzx is the error
        error -= pzx[t];
    }

    // convert deltas to normal from log scale
    apply(deltas, deltas, exp);

    return error / N;
}