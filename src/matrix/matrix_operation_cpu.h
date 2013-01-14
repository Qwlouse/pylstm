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

///Matrix multiplication
void mult(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out);

///Matrix multiplication and addition
void mult_add(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out);

///Elementwise multiplication
void dot(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out);

///Elementwise multiplication and add
void dot_add(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out);

///Apply sigmoid to all units
void apply_sigmoid(MatrixView2DCPU a, MatrixView2DCPU out);

///Apply tanh to all units
void apply_tanh(MatrixView2DCPU a, MatrixView2DCPU out);

///Apply tanh * 2 to all units
void apply_tanhx2(MatrixView2DCPU a, MatrixView2DCPU out);

bool equals(MatrixView2DCPU a, MatrixView2DCPU out);

void squash(MatrixView2DCPU a, MatrixView2DCPU b, d_type alpha);

#endif

