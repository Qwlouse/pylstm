/*
 * LstmLayer.h
 *
 *  Created on: Oct 27, 2010
 *      Author: koutnij
 */
#include <vector>
#include <math.h>

#include "Constants.h"
#include "LstmBlock.h"

#ifndef LSTMLAYER_H_
#define LSTMLAYER_H_

class LstmLayer {

public:
	LstmLayer(int nCells, int nInputs, int nOutputs, int nBlocks);
	void forwardPassStep(std::vector<LstmBlockState>& thisStates, std::vector<LstmBlockState>& prevStates, std::vector<prec>& input, bool isStartState);
	void backwardPassStep(std::vector<LstmBlockState>& thisStates, std::vector<LstmBlockState>& prevStates, std::vector<LstmBlockState>& nextStates,
		 std::vector< std::vector<prec> >& wK,
		 std::vector<prec>& dk, std::vector<prec>& input,
		 bool skipPrev, bool skipNext);
	void getBlockOutputs(std::vector<LstmBlockState>& states, bool skip, std::vector<prec>& bcs);
	void updateWeights(prec eta, prec alpha);
	void getSerializedWeights(std::vector<prec> & w);
	void setSerializedWeights(std::vector<prec>::iterator & it);
	void getSerializedDerivs(std::vector<prec> & w);
	void setSerializedDerivs(std::vector<prec>::iterator & it);
	void setConstantWeights(prec w);
	void setRandomWeights(prec halfRange);
	void resetDerivs();
	int nBlocks, nOutputs, nCells, nBxnC;

	~LstmLayer();
	std::vector <LstmBlock> lstmBlocks; // lstm blocks
private:

	std::vector<std::vector<std::vector<prec> > > wHtrans;
	std::vector<std::vector<std::vector<prec> > > wKtrans;
	std::vector<std::vector<std::vector<prec> > > gw;
	std::vector<prec> pbcs;
};

#endif /* LSTMLAYER_H_ */
