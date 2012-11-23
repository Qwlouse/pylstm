/*
 * ForwardLayer.cpp
 *
 *  Created on: Oct 27, 2010
 *      Author: koutnij
 */

#include <iostream>
#include "Constants.h"
#include "ForwardLayer.h"
#include "MatlabMath.h"

ForwardLayer::ForwardLayer(int nInputs, int nOutputs) {
	// nInputs = nBlocks * nCells;
	this->nInputs = nInputs;
	this->nOutputs = nOutputs;

	std::vector<prec> zeroI(nInputs, 0.);
	std::vector<prec> zeroO(nOutputs, 0.);

	wK.assign(nOutputs, zeroI);
	wKd.assign(nOutputs, zeroI);
	wKm.assign(nOutputs, zeroI);

	biasK  = zeroO;
	biasKd = zeroO;
	biasKm = zeroO;
}

void ForwardLayer::forwardPassStep(ForwardLayerState& thisState, std::vector<prec>& input){
	for(int i=0;i<nOutputs;i++){
		thisState.ak[i] = inner_product(&wK[i][0], &input[0], nInputs) + biasK[i];
		thisState.bk[i] = fnO(thisState.ak[i]);
	}
}

void ForwardLayer::backwardPassStep(ForwardLayerState& thisState, std::vector<prec>& input, std::vector<prec>& er){
	// vector_difference(tr[t].begin(), er[t].end(), state[t].bk.begin(),state[t].ek.begin() );// output error
	thisState.ek = er; // = copy operator

	for(int i=0; i<nOutputs; i++){
		thisState.dk[i] = fnOd(thisState.ak[i]) * thisState.ek[i];
		vector_add_scale(&wKd[i][0], &input[0], thisState.dk[i], nInputs);
		biasKd[i] += thisState.dk[i]; // bias treatment
	}
}

void ForwardLayer::setConstantWeights(prec w){
	wK.assign(nOutputs, std::vector<prec> (nInputs, w));
	biasK.assign(nOutputs, w);
}

void ForwardLayer::setRandomWeights(prec halfRange){
	for(int i=0;i<nOutputs;i++){
		for(int j=0;j<nInputs;j++){
			wK[i][j] = randomW(halfRange);
		}
		biasK[i] = randomW(halfRange);
	}
}

void ForwardLayer::updateWeights(prec eta, prec alpha){
	for(int i=0;i<nOutputs;i++){
		scalar_multiple(eta,   &wKd[i][0], nInputs); // d = eta * d
		scalar_multiple(alpha, &wKm[i][0], nInputs); // m = alpha * m
		vector_add(&wKm[i][0], &wKd[i][0], nInputs); // m = m + d
		vector_add(&wK[i][0],  &wKm[i][0], nInputs); // w = w + m
	}

	// bias treatment
	scalar_multiple(eta,   &biasKd[0], nOutputs);
	scalar_multiple(alpha, &biasKm[0], nOutputs);
	vector_add(&biasKm[0], &biasKd[0], nOutputs);
	vector_add(&biasK[0],  &biasKm[0], nOutputs);
}

void ForwardLayer::resetDerivs(){
	wKd.assign(nOutputs, std::vector <prec> (nInputs, 0.));
	biasKd.assign(nOutputs, 0.);
}

void ForwardLayer::getSerializedWeights(std::vector<prec> & w){
	//w.reserve(w.size() + (nOutputs * nInputs) + nOutputs);
	for(int i=0;i<nOutputs;i++){
		w.insert(w.end(), wK[i].begin(), wK[i].end());
	}
	w.insert(w.end(), biasK.begin(), biasK.end());
}

void ForwardLayer::setSerializedWeights(std::vector<prec>::iterator & it){
	for(int i=0;i<nOutputs;i++){
		wK[i].assign(it, it+nInputs); it += nInputs;
	}
	biasK.assign(it, it+nOutputs); it += nOutputs;
}

void ForwardLayer::getSerializedDerivs(std::vector<prec> & w){
	//w.reserve(w.size() + (2 * nOutputs * nInputs) + (2 * nOutputs));
	
	for(int i=0;i<nOutputs;i++){
		w.insert(w.end(), wKd[i].begin(), wKd[i].end());
		w.insert(w.end(), wKm[i].begin(), wKm[i].end());
	}
	w.insert(w.end(), biasKd.begin(), biasKd.end());
	w.insert(w.end(), biasKm.begin(), biasKm.end());
}

void ForwardLayer::setSerializedDerivs(std::vector<prec>::iterator & it){
	for(int i=0;i<nOutputs;i++){
		wKd[i].assign(it, it+nInputs); it += nInputs;
		wKm[i].assign(it, it+nInputs); it += nInputs;
	}
	biasKd.assign(it, it+nOutputs); it += nOutputs;
	biasKm.assign(it, it+nOutputs); it += nOutputs;
}

ForwardLayer::~ForwardLayer() {}
