#include "mex.h" /* Always include this */
#include "../experiment/Experiment.h"

// Matlab mex interface
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 3) {
        mexErrMsgTxt("mexLSTMgetNStates requires 3 input arguments: nOutputs, nCells, nBlocks");
    }
    
    // read input arguments (no checking)
    double* pnOutputs  = mxGetPr(prhs[0]);
    int nOutputs = (int) pnOutputs[0];
    double* pnCells  = mxGetPr(prhs[1]);
    int nCells = (int) pnCells[0];
    double* pnBlocks = mxGetPr(prhs[2]);
    int nBlocks = (int) pnBlocks[0];
    
    // get number of states
    int nStates = Experiment::getNSerializedStates(nOutputs, nCells, nBlocks);

    // output arguments
    plhs[0] = mxCreateDoubleScalar((double) nStates);
    return;
}