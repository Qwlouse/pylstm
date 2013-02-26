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

///Elementwise multiplication and add
void dot_add(Matrix a, Matrix b, Matrix out);

///Matrix multiplication
void mult(Matrix a, Matrix b, Matrix out, d_type scale = 1.0);

///Matrix multiplication and addition
void mult_add(Matrix a, Matrix b, Matrix out, d_type scale = 1.0);

///Apply sigmoid to all elements of a
void apply_sigmoid(Matrix a, Matrix out);

void apply_sigmoid_deriv(Matrix a, Matrix out);

///Apply tanh to all elements of a
void apply_tanh(Matrix a, Matrix out);

void apply_tanh_deriv(Matrix a, Matrix out);

///Apply tanh * 2 to all elements of a
void apply_tanhx2(Matrix a, Matrix out);

void apply_tanhx2_deriv(Matrix a, Matrix out);


//
/////Copy
//void copy(MatrixView2DCPU a, MatrixView2DCPU b);
//void copy(MatrixView3DCPU a, MatrixView3DCPU b);\
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



#endif

