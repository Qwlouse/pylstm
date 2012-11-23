#include "mex.h" /* Always include this */
#include "MexConverts.h"
#include <vector>
#include "../lstm/Constants.h"
#include "../experiment/Experiment.h"

// Matlab mex interface
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 12) { mexErrMsgTxt("mexLSTMtrain requires 12 input arguments: <nCells>, <nBlocks>, <serializedWeights>, <serializedDerivs>, <serializedStates>, <inputs>, <isStarts>, <targets>, <isTargets>, <learnRate>, <momentum> and <nEpochs>"); }
    
    // read input arguments (no checking)
    double* pnCells  = mxGetPr(prhs[0]);
    int nCells = (int) pnCells[0];
    double* pnBlocks = mxGetPr(prhs[1]);
    int nBlocks = (int) pnBlocks[0];
    std::vector<prec> serializedWeights = mxArray2v<prec>(prhs[2]);
    std::vector<prec> serializedDerivs = mxArray2v<prec>(prhs[3]);
	std::vector<prec> serializedStates = mxArray2v<prec>(prhs[4]);
    std::vector<std::vector<prec> > inputs = mxArray2vv<prec>(prhs[5]);
    std::vector<bool> isStarts = mxArray2v<bool>(prhs[6]);
    std::vector<std::vector<prec> > targets = mxArray2vv<prec>(prhs[7]);
    std::vector<bool> isTargets = mxArray2v<bool>(prhs[8]);
    double* pLearnRate = mxGetPr(prhs[9]);
    prec learnRate = (prec) pLearnRate[0];
    double* pMomentum = mxGetPr(prhs[10]);
    prec momentum = (prec) pMomentum[0];
    double* pnEpochs = mxGetPr(prhs[11]);
    int nEpochs = (int) pnEpochs[0];
    
    // some basic sanity checks
    int nSamples = inputs.size();
    if (nSamples != targets.size()) { mexErrMsgTxt("Number of input samples does not equal number of target samples"); }
    if (nSamples != isStarts.size()) { mexErrMsgTxt("Number of samples does not equal number of elements in <isStarts>"); }
    if (nSamples != isTargets.size()) { mexErrMsgTxt("Number of samples does not equal number of elements in <isTargets>"); }
    if (nSamples < 1) { mexErrMsgTxt("No samples in input data"); }
    int nInputs = inputs[0].size();
    int nOutputs = targets[0].size();
    
    // initialize and train
	bool finalpass = nlhs > 3; // only do final pass if outputs and errors are explicitly requested
    std::vector<prec> zeroOutVector = std::vector<prec>(nOutputs, (prec) 0.0);
	std::vector<std::vector<prec> > outputs(nSamples, zeroOutVector);
	Experiment* exper = new Experiment();
	exper->initLSTM(nInputs, nOutputs, nCells, nBlocks, serializedWeights);
    prec lasterr = exper->trainEpochs(serializedWeights, serializedDerivs, serializedStates, inputs, isStarts, targets, isTargets, outputs, finalpass, learnRate, momentum, nEpochs);
    delete exper;

    // set output arguments
    plhs[0] = v2mxArray<prec>(serializedWeights);
	if (nlhs > 1) { plhs[1] = v2mxArray<prec>(serializedDerivs); }
    if (nlhs > 2) { plhs[2] = v2mxArray<prec>(serializedStates); }
    if (nlhs > 3) { plhs[3] = mxCreateDoubleScalar(lasterr); } // errors after the last weight updates, computed with an additional forward pass
    if (nlhs > 4) { plhs[4] = vv2mxArray<prec>(outputs); } // outputs after the last weight updates, computed with an additional forward pass
    
    
    return;
}