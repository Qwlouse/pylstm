#include "mex.h" /* Always include this */
#include "../experiment/Experiment.h"

// Matlab mex interface
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 4) {
        mexErrMsgTxt("mexLSTMgetNDerivs requires 4 input arguments: nInputs, nOutputs, nCells, nBlocks");
    }
    
    // read input arguments (no checking)
    double* pnInputs  = mxGetPr(prhs[0]);
    int nInputs = (int) pnInputs[0];
    double* pnOutputs  = mxGetPr(prhs[1]);
    int nOutputs = (int) pnOutputs[0];
    double* pnCells  = mxGetPr(prhs[2]);
    int nCells = (int) pnCells[0];
    double* pnBlocks = mxGetPr(prhs[3]);
    int nBlocks = (int) pnBlocks[0];
    
    // get number of derivatives
	int nDerivs = LstmNetwork::getNSerializedDerivs(nInputs, nOutputs, nCells, nBlocks);

    // output arguments
    plhs[0] = mxCreateDoubleScalar((double) nDerivs);
    return;
}