#include <iostream>

#include "matrix.h"
#include "matrix_operation_cpu.h"

using namespace std;

int main(int argc, char **argv) {
  MatrixCPU a(10, 10, 1);
  MatrixCPU b(10, 10, 1);
  MatrixCPU c(10, 10, 1);
  mult(a, b, c);
  cout << c << endl;
}
