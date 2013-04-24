#ifndef __MATRIX_OPERATION_CPU_H__
#define __MATRIX_OPERATION_CPU_H__

#include "matrix.h"

#include <string>
#include <algorithm>
#include <numeric>
#include <math.h>
#include <blas.h>
#include <cstddef>


///Compare two matrices
bool equals(Matrix a, Matrix out);

///Elementwise add
//void add(Matrix a, Matrix b, Matrix out);

///Elementwise a+b into b
void add_into_b(Matrix a, Matrix b);

void add_vector_into(Matrix arg1, Matrix arg2);

///Add scalar b to every element in a
void add_scalar(Matrix a, d_type b);

///Elementwise multiplication
void dot(Matrix a, Matrix b, Matrix out);

///Elementwise multiplication
void dot_into_b(Matrix a, Matrix b);

///Elementwise multiplication and add
void dot_add(Matrix a, Matrix b, Matrix out);

///Matrix multiplication
void mult(Matrix a, Matrix b, Matrix out, d_type scale = 1.0);

///Matrix multiplication and addition
void mult_add(Matrix a, Matrix b, Matrix out, d_type scale = 1.0);



inline double sigmoid(double val) {
    return 1.0 / (1.0 + exp(-val));
}

inline double sigmoid_deriv(double val) {
    return val * (1.0 - val);
}

inline double tanhx2(double val) {
    return 2.0 * tanh(val);
}

inline double tanh_(double val) {
    return tanh(val);
}

inline double tanh_deriv(double val) {
    return 1.0 - val * val;
}

inline double tanhx2_deriv(double val) {
    return 2.0 - 0.5 * val * val;
}

inline double identity(double val) {
    return val;
}

inline double one(double) {
    return 1.0;
}


// Function pointer to a unary double function
typedef double (*unary_double_func)(double);


struct ActivationFunction {
	const unary_double_func f;
	const unary_double_func deriv;
	ActivationFunction(unary_double_func f = NULL, unary_double_func fp = NULL): f(f), deriv(fp) {};

	virtual void apply(Matrix a, Matrix out) const;
	virtual void apply_deriv(Matrix a, Matrix d, Matrix out) const;
};

struct SoftmaxLayerActivation: public ActivationFunction {
	SoftmaxLayerActivation() {};

	virtual void apply(Matrix a, Matrix out) const;
	virtual void apply_deriv(Matrix a, Matrix d, Matrix out) const;
};

const ActivationFunction Sigmoid(&sigmoid, &sigmoid_deriv);
const ActivationFunction Linear(&identity, &one);
const ActivationFunction Tanh(&tanh_, &tanh_deriv);
const SoftmaxLayerActivation Softmax;
const ActivationFunction Tanhx2(&tanhx2, &tanhx2_deriv);



void apply(Matrix in, Matrix out, unary_double_func f);

///Apply sigmoid to all elements of a
void apply_sigmoid(Matrix a, Matrix out);

void apply_sigmoid_deriv(Matrix a, Matrix out);

///Apply tanh to all elements of a
void apply_tanh(Matrix a, Matrix out);

void apply_tanh_deriv(Matrix a, Matrix out);

///Apply tanh * 2 to all elements of a
void apply_tanhx2(Matrix a, Matrix out);

void apply_tanhx2_deriv(Matrix a, Matrix out);



///squash
void squash(Matrix a, Matrix out);

void squash(Matrix a, Matrix out, d_type scale);


///Copy the content of a into b
void copy(Matrix a, Matrix b);

///Elementwise multiplication, with squash to size of out (out is smaller than a and b)
void dot_squash(Matrix a, Matrix b, Matrix out);

void dot_squash(Matrix a, Matrix b, Matrix out, d_type scale);

void scale_into(Matrix a, d_type alpha);

/////Elementwise multiplication and add, with squash to size of out (out is smaller than a and b)
//void dot_add_squash(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out, d_type const scale = 1.0);




#endif

