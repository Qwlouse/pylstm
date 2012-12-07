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

  MatrixCPU bla({{2,3,5},{3,4,55}});
  MatrixCPU bla2({{2,3,5},{3,4,55}});
  MatrixCPU bloe({{2,3,5},{3,4,54}});

  
  cout << bla << endl;
  cout << "should be true: " << equals(bla, bla2) << endl;
  cout << "should be false: " << equals(bla, bloe) << endl;
}
