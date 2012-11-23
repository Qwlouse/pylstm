#include "mex.h"
#include "blas.h"

// compile with: mex mexBLAS.cpp libmwblas.lib

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray 
*prhs[])
{
  double *A, *B, *C, one = 1.0, zero = 0.0;
  int m,n,p; 
  char *chn = "N";

  A = mxGetPr(prhs[0]);
  B = mxGetPr(prhs[1]);
  m = mxGetM(prhs[0]);
  p = mxGetN(prhs[0]);
  n = mxGetN(prhs[1]);

  if (p != mxGetM(prhs[1])) {
    mexErrMsgTxt("Inner dimensions of matrix multiply do not match");
  }

  plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
  C = mxGetPr(plhs[0]);

  /* Pass all arguments to Fortran by reference */
  dgemm (chn, chn, &m, &n, &p, &one, A, &m, B, &p, &zero, C, &m);
}
