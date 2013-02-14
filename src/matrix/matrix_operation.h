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

///Add scalar b to every element in a
//void add_scalar(MatrixView2DCPU a, d_type b);
//
/////Copy
//void copy(MatrixView2DCPU a, MatrixView2DCPU b);
//void copy(MatrixView3DCPU a, MatrixView3DCPU b);\
//
/////Matrix multiplication
//void mult(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out, d_type scale = 1.0);
//
/////Matrix multiplication and addition
//void mult_add(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out, d_type scale = 1.0);
//
/////Elementwise multiplication
//void dot(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out);
//
/////Elementwise multiplication and add
//void dot_add(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out);
//
/////Elementwise multiplication and add, with squash to size of out (out is smaller than a and b)
//void dot_add_squash(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out, d_type const scale = 1.0);
//
/////Elementwise multiplication, with squash to size of out (out is smaller than a and b)
//void dot_squash(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out, d_type const scale = 1.0);
//
/////squash
//void squash(MatrixView2DCPU a, MatrixView2DCPU out, d_type scale = 1.0);
//
/////Apply sigmoid to all units
//void apply_sigmoid(MatrixView2DCPU a, MatrixView2DCPU out);
//void apply_sigmoid(MatrixView3DCPU a, MatrixView3DCPU out);
//void apply_sigmoid_deriv(MatrixView2DCPU a, MatrixView2DCPU out);
//void apply_sigmoid_deriv(MatrixView3DCPU a, MatrixView3DCPU out);
//
/////Apply tanh to all units
//void apply_tanh(MatrixView2DCPU a, MatrixView2DCPU out);
//void apply_tanh(MatrixView3DCPU a, MatrixView3DCPU out);
//void apply_tanh_deriv(MatrixView2DCPU a, MatrixView2DCPU out);
//
/////Apply tanh * 2 to all units
//void apply_tanhx2(MatrixView2DCPU a, MatrixView2DCPU out);
//void apply_tanhx2(MatrixView2DCPU a, MatrixView2DCPU out);
//void apply_tanhx2_deriv(MatrixView2DCPU a, MatrixView2DCPU out);
//

//


#endif

