/*
 * LstmNetwork.cpp
 *
 *  Created on: Oct 27, 2010
 *      Author: koutnij
 */

#include <iostream>
#include "Constants.h"
#include "LstmNetwork.h"

LstmNetwork::LstmNetwork(int nInputs, int nOutputs, int nCells, int nBlocks) {
	nWeights = getNSerializedWeights(nInputs, nOutputs, nCells, nBlocks);
	nDerivs = nWeights * 2; // number of weights + momenta
	lstmLayer = new LstmLayer(nCells, nInputs, nOutputs, nBlocks);
	outputLayer = new ForwardLayer(nCells * nBlocks, nOutputs);
}

// forward computes forward pass without storing activations
void LstmNetwork::forward(int t1, int t2, std::vector<LstmBlockState>& lstmCurrentStates, ForwardLayerState& forwardCurrentState, 
	std::vector<std::vector<prec> >& inputs, std::vector<bool>& isStarts, std::vector<std::vector<prec>>& outputs){

	std::vector<LstmBlockState> prevStates = lstmCurrentStates;
	std::vector<LstmBlockState> emptyStates(lstmLayer->nBlocks);

	std::vector<prec> bcs;
	for(int t = t1; t < t2;t++){
		if (isStarts[t])
			prevStates = emptyStates;
		else
			prevStates = lstmCurrentStates; // copy operator

		lstmLayer->forwardPassStep(lstmCurrentStates, prevStates, inputs[t], isStarts[t]); // write values for this timestep
		lstmLayer->getBlockOutputs(lstmCurrentStates, false, bcs); // get all blocks outputs for this timestep
		outputLayer->forwardPassStep(forwardCurrentState, bcs); // write values for this timestep to outputLayerState

		outputs[t] = forwardCurrentState.bk;
	}
}

// forwardPass storing all activations
void LstmNetwork::forwardPass(int t1, int t2, std::vector<LstmBlockState>& lstmCurrentStates, ForwardLayerState& forwardCurrentState,
	    std::vector <std::vector<LstmBlockState>>& lstmLayerStates, std::vector<ForwardLayerState>& outputLayerStates,
		std::vector<std::vector<prec>>& inputs, std::vector<bool>& isStarts, std::vector<std::vector<prec>>& outputs){

	bool skipPrev;
	std::vector<LstmBlockState>* prevStates = &lstmCurrentStates;
	std::vector<LstmBlockState> emptyStates(lstmLayer->nBlocks);

	std::vector<prec> bcs;
	for(int t = t1; t<t2; t++){
		if (isStarts[t])
			prevStates = &emptyStates;
		else if (t > 0)
			prevStates = &lstmLayerStates[t-1];

		lstmLayer->forwardPassStep(lstmLayerStates[t], *prevStates, inputs[t], isStarts[t]);
		lstmLayer->getBlockOutputs(lstmLayerStates[t], false, bcs);
		outputLayer->forwardPassStep(outputLayerStates[t], bcs);
		outputs[t] = outputLayerStates[t].bk;
	}
	if (t2 > 0) {
		lstmCurrentStates = lstmLayerStates[t2-1];
		forwardCurrentState = outputLayerStates[t2-1];
	}

}

// t1 < t2
void LstmNetwork::backwardPass(int t1, int t2, std::vector<LstmBlockState>& lstmStartStates,
		std::vector <std::vector<LstmBlockState>>& lstmLayerStates, std::vector<ForwardLayerState>& outputLayerStates, 
		std::vector<std::vector<prec> >& errors, std::vector<std::vector<prec> >& inputs, std::vector<bool>& isStarts){

	bool skipPrev;
	bool skipNext;
	std::vector<LstmBlockState> *prevStates;
	std::vector<LstmBlockState> *nextStates;
	std::vector<LstmBlockState> emptyStates(lstmLayer->nBlocks);

	std::vector<prec> bcs;
	for(int t=t2-1;t>=t1;t--){
		skipPrev = (t <= 0) || isStarts[t];
		skipNext = (t >= (lstmLayerStates.size() - 1)) || isStarts[t+1];
		
		if (isStarts[t])
			prevStates = &emptyStates;
		else if (t > 0)
			prevStates = &lstmLayerStates[t-1];
		else
			prevStates = &lstmStartStates;

		if (skipNext)
			nextStates = &emptyStates;
		else
			nextStates = &lstmLayerStates[t+1];

		lstmLayer->getBlockOutputs(lstmLayerStates[t], false, bcs);
		outputLayer->backwardPassStep(outputLayerStates[t], bcs, errors[t]);
		lstmLayer->backwardPassStep(lstmLayerStates[t], *prevStates, *nextStates, outputLayer->wK, outputLayerStates[t].dk, inputs[t], skipPrev, skipNext); // was outputLayerState[t+1].bk - was wrong
	}
}

void LstmNetwork::updateWeights(prec eta, prec alpha){
	lstmLayer->updateWeights(eta, alpha);
	outputLayer->updateWeights(eta, alpha);
}

void LstmNetwork::resetDerivs(){
	lstmLayer->resetDerivs();
	outputLayer->resetDerivs();
}

int LstmNetwork::getNSerializedWeights(int nInputs, int nOutputs, int nCells, int nBlocks) {
	int nBxnC = nBlocks * nCells;
	return nBlocks * ( (3 * nInputs) + (3 * nCells) + (nCells * nInputs) +	// LSTM weights1
		               (3 * nBxnC)   + (nCells * nBxnC) +					// LSTM weights2
		                3 + nCells											// LSTM bias
					 ) +
		   (nOutputs * nBxnC) + nOutputs;									// ForwardLayer
}

int LstmNetwork::getNSerializedDerivs(int nInputs, int nOutputs, int nCells, int nBlocks) {
	return 2 * getNSerializedWeights(nInputs, nOutputs, nCells, nBlocks);
}


void LstmNetwork::getSerializedWeights(std::vector<prec>& w){
	w.reserve(nWeights);
	lstmLayer->getSerializedWeights(w);
	outputLayer->getSerializedWeights(w);
}

void LstmNetwork::setSerializedWeights(std::vector<prec>& w){
	if (w.size() != nWeights) {
		std::cerr << "Number of weights (" << w.size() << ") incorrect! Correct number is " << nWeights;
		throw "number of weights is incorrect!";
	}
	std::vector<prec>::iterator it = w.begin();
	lstmLayer->setSerializedWeights(it);
	outputLayer->setSerializedWeights(it);
}

void LstmNetwork::getSerializedDerivs(std::vector<prec>& w){
	w.reserve(nDerivs);
	lstmLayer->getSerializedDerivs(w);
	outputLayer->getSerializedDerivs(w);
}

void LstmNetwork::setSerializedDerivs(std::vector<prec>& w){
	if (w.size() != nDerivs) {
		std::cerr << "Number of derivatives (" << w.size() << ") incorrect! Correct number is " << nDerivs;
		throw "number of derivatives is incorrect!";
	}
	std::vector<prec>::iterator it = w.begin();
	lstmLayer->setSerializedDerivs(it);
	outputLayer->setSerializedDerivs(it);
}

void LstmNetwork::setConstantWeights(prec w){
	lstmLayer->setConstantWeights(w);
	outputLayer->setConstantWeights(w);
}

void LstmNetwork::setRandomWeights(prec halfRange){
	lstmLayer->setRandomWeights(halfRange);
	outputLayer->setRandomWeights(halfRange);
}

LstmNetwork::~LstmNetwork() {
	delete lstmLayer;
	delete outputLayer;
}
