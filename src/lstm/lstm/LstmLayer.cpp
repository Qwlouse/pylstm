/*
 * LstmLayer.cpp
 *
 *  Created on: Oct 27, 2010
 *      Author: koutnij
 */
#include <vector>
#include <math.h>
#include <iostream>

#include "LstmLayer.h"
#include "LstmBlock.h"
#include "Constants.h"

LstmLayer::LstmLayer(int nCells, int nInputs, int nOutputs, int nBlocks) {
	this->nCells = nCells;
	this->nBlocks = nBlocks;
	this->nOutputs = nOutputs;
	this->nBxnC = nBlocks * nCells;
	wKtrans.assign(nBlocks, std::vector<std::vector<prec> > (nCells, std::vector<prec> (nOutputs, 0.)));
	wHtrans.assign(nBlocks, std::vector<std::vector<prec> > (nCells, std::vector<prec> (nBxnC, 0.)));

	gw.assign(nBlocks, std::vector<std::vector<prec> > (nCells, std::vector<prec> (nBlocks * 3, 0.)));
	pbcs.assign(nBxnC, 0.);

	lstmBlocks.assign(nBlocks, LstmBlock(nCells, nInputs, nOutputs, nBlocks));
}

void LstmLayer::getBlockOutputs(std::vector<LstmBlockState>& states, bool skip, std::vector<prec>& bcs) {
	bcs.resize(nBxnC);
	if (skip) {
		bcs.assign(nBxnC, 0.);
	} else{
		for (int i=0;i<nBlocks;i++){
			for (int j=0;j<nCells;j++){
				bcs[i * nCells + j] = states[i].bc[j];
			}
		}
	}
}

void LstmLayer::forwardPassStep(std::vector<LstmBlockState>& thisStates, std::vector<LstmBlockState>& prevStates, std::vector<prec>& input, bool skipPrev){
	
	// output of all blocks for previous step, except in case the current state is a start state
	std::vector<prec> pbcs;
	getBlockOutputs(prevStates, skipPrev, pbcs);

	for (int i=0;i<nBlocks;i++){
		lstmBlocks[i].forwardPassStep(thisStates[i], prevStates[i], input, skipPrev, pbcs);
	}
}

void LstmLayer::backwardPassStep(std::vector<LstmBlockState>& thisStates, std::vector<LstmBlockState>& prevStates, std::vector<LstmBlockState>& nextStates,
		 std::vector< std::vector<prec> >& wK,
		 std::vector<prec>& dk, std::vector<prec>& input,
		 bool skipPrev, bool skipNext){

	// transpose wK and wH
	for(int i=0;i<nBlocks;i++){
		for(int j=0;j<nCells;j++){
			for(int k=0;k<nOutputs;k++){
				wKtrans[i][j][k] = wK[k][i * nCells + j];
			}
		}
	}

	for(int i=0;i<nBlocks;i++){
		for(int j=0;j<nCells;j++){
			for(int k=0;k<nBxnC;k++){
				wHtrans[i][j][k] = lstmBlocks[k / nCells].wHc[k % nCells][i * nCells + j];
			}
		}
	}

	// concatenate states[t+1].dc, states[t+1].di,df,d_o, etc..
    std::vector<prec> db(nBxnC, 0.);
	//if(t < states[0].size() - 1){
	if(!skipNext) {
		for(int i=0;i<nBxnC;i++){
			db[i]=nextStates[i / nCells].dc[i % nCells];
		}
	}

	std::vector<prec> gd(nBlocks * 3, 0.);
	if(!skipNext) {
		for(int i=0;i<nBlocks;i++){
			gd[3 * i] = nextStates[i].di;
			gd[3 * i + 1] = nextStates[i].df;
			gd[3 * i + 2] = nextStates[i].d_o;
		}
	}

	for(int i=0;i<nBlocks;i++){
		for(int j=0;j<nCells;j++){
			for(int k=0;k<nBlocks;k++){
				gw[i][j][3 * k] = lstmBlocks[k].wHi[nCells * i + j];
				gw[i][j][3 * k+1] = lstmBlocks[k].wHf[nCells * i + j];;
				gw[i][j][3 * k+2] = lstmBlocks[k].wHo[nCells * i + j];;
			}
		}
	}

	// other block cells activations (needed for weight update)
	// except in case the current state <t> is a start state
	//std::vector<prec> pbcs;
	//getBlockOutputs(t-1, states, isStarts[t], pbcs); // QUESTION: jan said: 'was t-1', but is still t-1
	getBlockOutputs(prevStates, skipPrev, pbcs); // QUESTION: jan said: 'was t-1', but is still t-1

	for (int i=0;i<nBlocks;i++){
		lstmBlocks[i].backwardPassStep(thisStates[i], prevStates[i], nextStates[i], wKtrans[i], dk, wHtrans[i], db, gw[i], gd, input, skipPrev, skipNext, pbcs);  //QUESTION (jan said: 'fix')
	}
}

void LstmLayer::updateWeights(prec eta, prec alpha){
	for(int i=0;i<nBlocks;i++){
		lstmBlocks[i].updateWeights(eta, alpha);
	}
}

void LstmLayer::resetDerivs(){
	for(int i=0;i<nBlocks;i++){
		lstmBlocks[i].resetDerivs();
	}
}

void LstmLayer::getSerializedWeights(std::vector<prec> & w) {
	for(int i=0;i<nBlocks;i++){
		lstmBlocks[i].getSerializedWeights1(w);
	}
	for(int i=0;i<nBlocks;i++){
		lstmBlocks[i].getSerializedWeights2(w);
	}
}

void LstmLayer::setSerializedWeights(std::vector<prec>::iterator & it){
	for(int i=0;i<nBlocks;i++){
		lstmBlocks[i].setSerializedWeights1(it);
	}
	for(int i=0;i<nBlocks;i++){
		lstmBlocks[i].setSerializedWeights2(it);
	}
}

void LstmLayer::getSerializedDerivs(std::vector<prec> & w) {
	for(int i=0;i<nBlocks;i++){
		lstmBlocks[i].getSerializedDerivs1(w);
	}
	for(int i=0;i<nBlocks;i++){
		lstmBlocks[i].getSerializedDerivs2(w);
	}
}

void LstmLayer::setSerializedDerivs(std::vector<prec>::iterator & it){
	for(int i=0;i<nBlocks;i++){
		lstmBlocks[i].setSerializedDerivs1(it);
	}
	for(int i=0;i<nBlocks;i++){
		lstmBlocks[i].setSerializedDerivs2(it);
	}
}

void LstmLayer::setConstantWeights(prec w){
	for(int i=0;i<nBlocks;i++){
		lstmBlocks[i].setConstantWeights(w);
	}
}

void LstmLayer::setRandomWeights(prec halfRange){
	for(int i=0;i<nBlocks;i++){
		lstmBlocks[i].setRandomWeights(halfRange);
	}
}

LstmLayer::~LstmLayer() {

}
