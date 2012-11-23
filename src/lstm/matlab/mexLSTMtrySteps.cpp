#include "mex.h" /* Always include this */
#include "MexConverts.h"
#include <vector>
#include "../lstm/Constants.h"
#include "../experiment/Experiment.h"

// Matlab mex interface
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 8) {
        mexErrMsgTxt("mexLSTMtrySteps requires 8 input arguments: <nInputs>, <nOutputs>, <nCells>, <nBlocks>, <serializedWeights>, <serializedStates>, <inputs> and <isStarts>");
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
    std::vector<std::vector<prec> > inputs = mxArray2vv<prec>(prhs[6]);
    std::vector<bool> isStarts = mxArray2v<bool>(prhs[7]);
    
    // some basic sanity checks
    size_t nSamples = inputs.size();
    if (nSamples != isStarts.size()) { mexErrMsgTxt("Number of samples does not equal number of elements in <isStarts>"); }
    if (nSamples < 1) { mexErrMsgTxt("No samples in input data"); }
    
    // try a number of samples without updating the actual state of the network
	std::vector<prec> zeroOutVector = std::vector<prec>(nOutputs, (prec) 0.0);
	std::vector<std::vector<prec> > outputs(nSamples, zeroOutVector);
    std::vector<prec> startStates(serializedStates.size(), (prec) 0.0);
    Experiment* exper = new Experiment();
	exper->initLSTM(nInputs, nOutputs, nCells, nBlocks, serializedWeights);

	for (size_t i=0; i<nSamples; i++) {
        startStates = serializedStates; // copy serializedStates into startStates
		exper->forwardStep(startStates, inputs[i], isStarts[i], outputs[i]); // start again from startStates
    }

    delete exper;
    
    // set output arguments
    plhs[0] = vv2mxArray<prec>(outputs);
    return;
}