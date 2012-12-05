/*
 * matrix.cc
 *
 *  Created on: Sep 5, 2011
 *      Author: stollenga
 */

#include "matrix.h"
#include "matrix_operations.h"

Matrix::Matrix(MatrixPtr &from, bool do_copy) :
	d_data(from.size()),
	d_rows(from.rows()),
	d_columns(from.columns())
{
	if (do_copy)
		copy(from, *this);
}
