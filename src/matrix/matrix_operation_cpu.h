#ifndef __MATRIX_OPERATION_CPU_H__
#define __MATRIX_OPERATION_CPU_H__

#include "matrix_cpu.h"

#include <string>
#include <algorithm>
#include <numeric>
#include <math.h>
#include <blas.h>
#include <cstddef>

///Elementwise add
void add(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out);

///Elementwise a+b into b
void add_into_b(MatrixView2DCPU a, MatrixView2DCPU b);

///Add scalar b to every element in a
void add_scalar(MatrixView2DCPU a, d_type b);

void add_vector_into(MatrixView2DCPU arg1, MatrixView2DCPU arg2);

///Copy
void copy(MatrixView2DCPU a, MatrixView2DCPU b);
void copy(MatrixView3DCPU a, MatrixView3DCPU b);

///Matrix multiplication
void mult(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out, d_type scale = 1.0);

///Matrix multiplication and addition
void mult_add(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out, d_type scale = 1.0);

///Elementwise multiplication
void dot(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out);

///Elementwise multiplication and add
void dot_add(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out);

///Elementwise multiplication and add, with squash to size of out (out is smaller than a and b)
void dot_add_squash(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out, d_type const scale = 1.0);

///Elementwise multiplication, with squash to size of out (out is smaller than a and b)
void dot_squash(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out, d_type const scale = 1.0);

///squash
void squash(MatrixView2DCPU a, MatrixView2DCPU out, d_type scale = 1.0);

//scale values by alpha
void scale_into(MatrixView2DCPU arg1, d_type alpha);

///Apply sigmoid to all units
void apply_sigmoid(MatrixView2DCPU a, MatrixView2DCPU out);
void apply_sigmoid(MatrixView3DCPU a, MatrixView3DCPU out);
void apply_sigmoid_deriv(MatrixView2DCPU a, MatrixView2DCPU out);
void apply_sigmoid_deriv(MatrixView3DCPU a, MatrixView3DCPU out);

///Apply tanh to all units
void apply_tanh(MatrixView2DCPU a, MatrixView2DCPU out);
void apply_tanh(MatrixView3DCPU a, MatrixView3DCPU out);
void apply_tanh_deriv(MatrixView2DCPU a, MatrixView2DCPU out);

///Apply tanh * 2 to all units
void apply_tanhx2(MatrixView2DCPU a, MatrixView2DCPU out);
void apply_tanhx2(MatrixView2DCPU a, MatrixView2DCPU out);
void apply_tanhx2_deriv(MatrixView2DCPU a, MatrixView2DCPU out);

/// Apply softmax function
void apply_softmax(MatrixView2DCPU arg1, MatrixView2DCPU arg2);
void softmax_deriv(MatrixView2DCPU in_deltas, MatrixView2DCPU activations, MatrixView2DCPU activation_tmp, MatrixView2DCPU deltas);

///check if a==out
bool equals(MatrixView2DCPU a, MatrixView2DCPU out);



#endif

