/**
 * This header provides a layer of abstraction around the cblas interface.
 * This is needed because we allow using the matlab blas library which has a
 * different c interface than the standardized cblas interface.
 *
 * So if the MATLAB_BLAS variable is not set we just include cblas.h.
 * If, on the other hand, we use MATLAB_BLAS we provide the standard CBLAS
 * functions that under the hood call the matlab-cblas functions with their
 * weird signature (suckers!).
 */
#pragma once

#include "Config.h"
#include "core/Assert.h"

#ifndef USE_MATLAB_BLAS
// if we are not using matlab-blas we just include cblas.h  (easy)
#include "cblas.h"

#else // we import the matlab cblas and convert the interface (ugly!)
#include <blas.h>

// CBLAS types
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};


// Matlab BLAS constants
static char TRANS = 'T';
static char NO_TRANS = 'N';



void cblas_daxpy(const int N, const double alpha, const double *X,
                 const int incX, double *Y, const int incY) {

    ptrdiff_t n = N;
    ptrdiff_t inc_x = incX;
    ptrdiff_t inc_y = incY;

    daxpy(&n,
          const_cast<double*>(&alpha),
          const_cast<double*>(X),
          &inc_x,
          Y,
          &inc_y);
}


void cblas_dgbmv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const double alpha,
                 const double *A, const int lda, const double *X,
                 const int incX, const double beta, double *Y, const int incY) {
    ASSERT(order == CblasColMajor);
    char* trans_A = TransA == CblasTrans? &TRANS : &NO_TRANS;

    ptrdiff_t m = M;
    ptrdiff_t n = N;
    ptrdiff_t kl = KL;
    ptrdiff_t ku = KU;
    ptrdiff_t inc_x = incX;
    ptrdiff_t inc_y = incY;
    ptrdiff_t lda_ = lda;

    dgbmv(trans_A, &m, &n, &kl, &ku, const_cast<double*>(&alpha),
          const_cast<double*>(A), &lda_, const_cast<double*>(X),
          &inc_x, const_cast<double*>(&beta), Y, &inc_y);
}

void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc) {
    ASSERT(Order == CblasColMajor);
    char* trans_A = TransA == CblasTrans? &TRANS : &NO_TRANS;
    char* trans_B = TransB == CblasTrans? &TRANS : &NO_TRANS;
    ptrdiff_t m = M;
    ptrdiff_t n = N;
    ptrdiff_t k = K;
    ptrdiff_t lda_ = lda;
    ptrdiff_t ldb_ = ldb;
    ptrdiff_t ldc_ = ldc;

    dgemm(trans_A, trans_B, &m, &n, &k, const_cast<double*>(&alpha),
          const_cast<double*>(A), &lda_, const_cast<double*>(B), &ldb_,
          const_cast<double*>(&beta), C, &ldc_);
}

void cblas_dcopy(const int N, const double *X, const int incX,
                 double *Y, const int incY) {
    ptrdiff_t n = N;
    ptrdiff_t inc_x = incX;
    ptrdiff_t inc_y = incY;

    dcopy(&n, const_cast<double*>(X), &inc_x, Y, &inc_y);
}

void cblas_dscal(const int N, const double alpha, double *X, const int incX) {
    ptrdiff_t n = N;
    ptrdiff_t inc_x = incX;

    dscal(&n, const_cast<double*>(&alpha), X, &inc_x);
}

#endif