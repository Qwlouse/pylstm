/*
* LstmNetwork.h
*
*  Created on: Oct 27, 2010
*      Author: koutnij
*/

#include <vector>

#include "Constants.h"
#include "LstmBlock.h"
#include "LstmLayer.h"
#include "ForwardLayer.h"

#ifndef LSTMNETWORK_H_
#define LSTMNETWORK_H_

class LstmNetwork {
public:
	LstmNetwork(int nInputs, int nOutputs, int nCells, int nBlocks);
	void forward(int t1, int t2, std::vector<LstmBlockState>& lstmCurrentStates, ForwardLayerState& forwardCurrentState, 
		std::vector<std::vector<prec> >& inputs, std::vector<bool>& isStarts, std::vector<std::vector<prec>>& outputs);
	void forwardPass(int t1, int t2, std::vector<LstmBlockState>& lstmCurrentStates, ForwardLayerState& forwardCurrentState,
	    std::vector <std::vector<LstmBlockState>>& lstmLayerStates, std::vector<ForwardLayerState>& outputLayerStates,
		std::vector<std::vector<prec>>& inputs, std::vector<bool>& isStarts, std::vector<std::vector<prec>>& outputs);
	void backwardPass(int t1, int t2, std::vector<LstmBlockState>& lstmStartStates,
		std::vector <std::vector<LstmBlockState>>& lstmLayerStates, std::vector<ForwardLayerState>& outputLayerStates, 
		std::vector<std::vector<prec> >& errors, std::vector<std::vector<prec> >& inputs, std::vector<bool>& isStarts);
	void updateWeights(prec eta, prec alpha);
	void resetDerivs();
	void setConstantWeights(prec w);
	void setRandomWeights(prec halfRange);
	void setSerializedWeights(std::vector<prec>& w);
	void getSerializedWeights(std::vector<prec>& w);
	void setSerializedDerivs(std::vector<prec>& w);
	void getSerializedDerivs(std::vector<prec>& w);
	static int getNSerializedWeights(int nInputs, int nOutputs, int nCells, int nBlocks);
	static int getNSerializedDerivs(int nInputs, int nOutputs, int nCells, int nBlocks);

	~LstmNetwork();

	int nWeights, nDerivs;
	LstmLayer* lstmLayer;
	ForwardLayer* outputLayer;

private:

};

#endif /* LSTMNETWORK_H_ */
