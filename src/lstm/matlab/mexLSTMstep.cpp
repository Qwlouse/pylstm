#include "mex.h" /* Always include this */
#include "MexConverts.h"
#include <vector>
#include "../lstm/Constants.h"
#include "../experiment/Experiment.h"

// Matlab mex interface
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 8) {
        mexErrMsgTxt("mexLSTMstep requires 8 input arguments: nInputs, nOutputs, nCells, nBlocks, serializedWeights, serializedStates, input and isStart");
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
    std::vector<prec> serializedWeights = mxArray2v<prec>(prhs[4]);
    std::vector<prec> serializedStates = mxArray2v<prec>(prhs[5]);
    std::vector<prec> input = mxArray2v<prec>(prhs[6]);
    double* pIsStart  = mxGetPr(prhs[7]);
    bool isStart = (bool) pIsStart[0];
    std::vector<prec> output(nOutputs, 0.);

    // initialize and run one step
	Experiment* exper = new Experiment();
	exper->initLSTM(nInputs, nOutputs, nCells, nBlocks, serializedWeights);
	exper->forwardStep(serializedStates, input, isStart, output);
    delete exper;

    // set output arguments
    plhs[0] = v2mxArray<prec>(output);
    if (nlhs > 1) {plhs[1] = v2mxArray<prec>(serializedStates); };
    return;
}