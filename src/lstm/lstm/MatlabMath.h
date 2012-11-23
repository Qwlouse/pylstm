/*
 * Based on VectorMath.h
 *
 *  Created on: May, 2011
 *      Author: leo_pape
 */

#ifndef MATLABMATH_H_
#define MATLABMATH_H_

/* apparently, this is not true anymore
#if !defined(_WIN32)
#define dgemm dgemm_
#define dgbmv dgbmv_
#define dcopy dcopy_
#define daxpy daxpy_
#define ddot ddot_
#define dscal dscal_
#define sgemm sgemm_
#define sgbmv sgbmv_
#define scopy scopy_
#define saxpy saxpy_
#define sdot sdot_
#define sscal sscal_
#endif
*/

#include <cstdlib>
#include "blas.h"

static double d_p_one = 1.0, d_n_one = -1.0, d_zero = 0.0;
static double *pd_p_one  = &d_p_one, *pd_n_one = &d_n_one, *pd_zero = &d_zero; // for vector addition and subtraction operations
static float  f_p_one = 1.0, f_n_one = -1.0, f_zero = 0.0;
static float  *pf_p_one  = &f_p_one, *pf_n_one = &f_n_one, *pf_zero = &f_zero; // for vector addition and subtraction operations

static ptrdiff_t v_one_t  = 1, v_zero_t = 0;
static ptrdiff_t *one_t   = &v_one_t, *zero_t = &v_zero_t;    // increments for BLAS functions
static char v_N = 'N';
static char *N = &v_N;

// vector difference: z = x - y; x and y are left unchanged
static double* vector_difference ( double *x, double *y, double *z, ptrdiff_t n) {
	dcopy(&n, x, one_t, z, one_t); // z = x with copy operation
	daxpy(&n, pd_n_one, y, one_t, z, one_t); // z = (-y + z)
	return z;
}
// vector difference: z = x - y; x and y are left unchanged
static float* vector_difference ( float *x, float *y, float *z, ptrdiff_t n) {
	scopy(&n, x, one_t, z, one_t); // z = x with copy operation
	saxpy(&n, pf_n_one, y, one_t, z, one_t); // z = (-y + z)
	return z;
}


//NOT TESTED vector elementwise multiplication: z = x .* y;
static double* vector_melt ( double *x, double *y, double *z, ptrdiff_t n) {
	dgbmv(N, &n, &n, zero_t, zero_t, pd_p_one, x, &n, y, one_t, pd_zero, z, one_t);
	return z;
}
//NOT TESTED vector elementwise multiplication: z = x .* y;
static float* vector_melt ( float *x, float *y, float *z, ptrdiff_t n) {
	sgbmv(N, &n, &n, zero_t, zero_t, pf_p_one, x, &n, y, one_t, pf_zero, z, one_t);
	return z;
}


// vector add: x = x + y
static double* vector_add ( double *x, double *y, ptrdiff_t n) {
    daxpy(&n, pd_p_one, y, one_t, x, one_t); // x = x + y
    return x;
}
// vector add: x = x + y
static float* vector_add ( float *x, float *y, ptrdiff_t n) {
    saxpy(&n, pf_p_one, y, one_t, x, one_t); // x = x + y
    return x;
}


// add scaled vector to original vector: x = x + ay
static double* vector_add_scale ( double *x, double *y, double k, ptrdiff_t n) {
    daxpy(&n, &k, y, one_t, x, one_t); // x = x + ay
    return x;
}
// add scaled vector to original vector: x = x + ay
static float* vector_add_scale ( float *x, float *y, float k, ptrdiff_t n) {
    saxpy(&n, &k, y, one_t, x, one_t); // x = x + ay
    return x;
}


// scale vector: x = kx
static double* scalar_multiple (double k, double *x, ptrdiff_t n) {
	dscal(&n, &k, x, one_t);
	return x;
}
// scale vector: x = kx
static float* scalar_multiple (float k, float *x, ptrdiff_t n) {
	sscal(&n, &k, x, one_t);
	return x;
}


// scale vector: y = kx
static double* scalar_multiple (double k, double *x, double *y, ptrdiff_t n) {
	dcopy(&n, x, one_t, y, one_t); // y = x with copy operation
	dscal(&n, &k, y, one_t);
	return y;
}
// scale vector: y = kx
static float* scalar_multiple (float k, float *x, float *y, ptrdiff_t n) {
	scopy(&n, x, one_t, y, one_t); // y = x with copy operation
	sscal(&n, &k, y, one_t);
	return y;
}


// sum of squared error: sse = t + sum(x .* x)
static double inner_product(double *x, double *y, ptrdiff_t n) {
	return ddot(&n, x, one_t, y, one_t);
}
// sum of squared error: sse = t + sum(x .* x)
static float inner_product(float *x, float *y, ptrdiff_t n) {
	return sdot(&n, x, one_t, y, one_t);
}


// sum of squared error: sse = sum(x .* x)
static double sse(double *x, ptrdiff_t n) {
	return ddot(&n, x, one_t, x, one_t);
}
// sum of squared error: sse = sum(x .* x)
static float sse(float *x, ptrdiff_t n) {
	return sdot(&n, x, one_t, x, one_t);
}

template <class T>
		static T randomW(T halfRange){
	return halfRange * (2 *  (rand()/(T)RAND_MAX) -1 );
}


#endif /* MATLABMATH_H_ */
