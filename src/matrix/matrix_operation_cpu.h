#ifndef __MATRIX_OPERATION_CPU_H__
#define __MATRIX_OPERATION_CPU_H__

#include "matrix_cpu.h"

#include <string>
#include <algorithm>
#include <numeric>
#include <math.h>
#include <blas.h>
#include <cstddef>

static d_type double_one = 1.0;
static d_type double_zero = 0.0;
static d_type double_min_one = -1.0;
static char NO_TRANS = 'N';
static char TRANS = 'T';
static ptrdiff_t diff_one = 1;
static ptrdiff_t diff_zero = 0;

void mult(MatrixView2DCPU &a, MatrixView2DCPU &b, MatrixView2DCPU &out);

#endif

