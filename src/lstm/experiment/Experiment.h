/*
 * Experiment.h
 *
 *  Created on: Nov 15, 2010
 *      Author: koutnij
 */

#ifndef EXPERIMENT_H_
#define EXPERIMENT_H_


#include <vector>
#include <iostream>
#include <string>

#include "Experiment.h"
#include "../lstm/Constants.h"
#include "../lstm/LstmLayer.h"
#include "../lstm/ForwardLayer.h"
#include "../lstm/LstmNetwork.h"
#include "../lstm/MatlabMath.h"

class Experiment {
public:
	std::vector< std::vector<LstmBlockState> > lstmStateBuffer;
	std::vector<ForwardLayerState> outputLayerStateBuffer;
	std::vector<std::vector<prec>> errorBuffer;

	int nCells, nBlocks, nInputs, nOutputs, nStates;
	int buffersLength;

	LstmNetwork* lstmNetwork;

	Experiment();

	//int nCells=1,nBlocks=2,nInputs=3,nOutputs=3,nX=10,dimX=3;
	//void initializeOneStepPrediction(int nCells, int nBlocks, std::vector<std::vector <prec> >& dataSequence, std::vector<std::vector<prec>> inputSequenceBuffer,
	//std::vector<std::vector<prec>> trainingSequenceBuffer, std::vector<bool> isTargetBuffer, std::vector<bool> isStartBuffer);
	void forward(std::vector<prec>& serializedStates, 
				 std::vector<std::vector<prec>>&inputs, std::vector<bool>& isStarts, 
				 std::vector<std::vector<prec>>& outputs);
	void forwardStep(std::vector<prec>& serializedStates, std::vector<prec>& input, bool isStart, std::vector<prec>& output);
	void forwardSteps(std::vector<prec>& serializedStates, std::vector<std::vector<prec> >& inputs,  
					  std::vector<bool>& isStarts, std::vector<std::vector<prec> >& outputs);
	prec trainEpochs(std::vector<prec>& serializedWeights, std::vector<prec>& serializedDerivs,
					 std::vector<prec>& serializedStates, std::vector<std::vector<prec> >& inputs,
					 std::vector<bool>& isStarts, std::vector<std::vector<prec>>& targets, std::vector<bool>& isTargets,
					 std::vector<std::vector<prec> >& outputs, bool finalpass, prec learningRate, prec momentum, int nEpochs);
	prec getOutputError(std::vector<std::vector<prec>>& outputs, 
						std::vector<std::vector<prec>>& targets, std::vector<bool>& isTargets, 
						std::vector<std::vector<prec> >& errors);
	void setSerializedStates(std::vector<LstmBlockState>& lstmStates, ForwardLayerState& forwardState, std::vector<prec> &s);
	void getSerializedStates(std::vector<LstmBlockState>& lstmStates, ForwardLayerState& forwardState, std::vector<prec> &s);
	void initLSTM(int nInputs, int nOutputs, int nCells, int nBlocks, std::vector<prec> serializedWeights);
	static void readMatrix(std::vector<std::vector<prec> >& mat, int lineLength);
	static int getNSerializedStates(int nOutputs, int nCells, int nBlocks);
	~Experiment();


};

#endif /* EXPERIMENT_H_ */
