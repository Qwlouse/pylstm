#include "matrix_operation_cpu.h"
#include <iostream>

using namespace std;


void mult(MatrixView2DCPU &a, MatrixView2DCPU &b, MatrixView2DCPU &out) {
	char a_state = (a.matrix_state == NORMAL) ? 'N' : 'T';
	char b_state = (b.matrix_state == NORMAL) ? 'N' : 'T';
    
    cout << a_state << " " << b_state << " " << a.n_rows << " " << b.n_rows << " " << a.n_columns << " " << b.n_columns << " " << a.data << " " << b.data << endl;

	dgemm(&a_state, &b_state, &a.n_rows, &b.n_columns, &a.n_columns,
		&double_one,
		a.data,
		&a.n_rows, b.data, &b.n_rows, &double_zero, out.data, &out.n_rows);
}


void mult_add(MatrixView2DCPU &a, MatrixView2DCPU &b, MatrixView2DCPU &out) {
	char a_state = (a.matrix_state == NORMAL) ? 'N' : 'T';
	char b_state = (b.matrix_state == NORMAL) ? 'N' : 'T';
    
    cout << a_state << " " << b_state << " " << a.n_rows << " " << b.n_rows << " " << a.n_columns << " " << b.n_columns << " " << a.data << " " << b.data << endl;

	dgemm(&a_state, &b_state, &a.n_rows, &b.n_columns, &a.n_columns,
		&double_one,
		a.data,
		&a.n_rows, b.data, &b.n_rows, &double_one, out.data, &out.n_rows);
}
