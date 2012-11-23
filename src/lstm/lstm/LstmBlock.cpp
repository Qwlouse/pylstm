/*
 * LstmBlock.cpp
 *
 *  Created on: Oct 26, 2010
 *      Author: koutnij
 */
#include <vector>
#include <iostream>

#include "Constants.h"
#include "LstmBlock.h"
#include "MatlabMath.h"

LstmBlock::LstmBlock(int nCells, int nInputs, int nOutputs, int nBlocks){
this->nCells = nCells;
	this->nInputs = nInputs;
	this->nBlocks = nBlocks;
	this->nOutputs = nOutputs;
	this->nBxnC = nBlocks * nCells;

	std::vector<prec> zeroI (nInputs, 0.);
	std::vector<prec> zeroH (nBxnC, 0.);
	std::vector<prec> zeroC (nCells, 0.);

	// input gate weights
	wIi = zeroI;
	wHi = zeroH;
	wCi = zeroC;
	wIid = zeroI;
	wHid = zeroH;
	wCid = zeroC;
	wIim = zeroI;
	wHim = zeroH;
	wCim = zeroC;

	// forget gate weights
	wIf = zeroI;
	wHf = zeroH;
	wCf = zeroC;
	wIfd = zeroI;
	wHfd = zeroH;
	wCfd = zeroC;
	wIfm = zeroI;
	wHfm = zeroH;
	wCfm = zeroC;

	// output gate weights
	wIo = zeroI;
	wHo = zeroH;
	wCo = zeroC;
	wIod = zeroI;
	wHod = zeroH;
	wCod = zeroC;
	wIom = zeroI;
	wHom = zeroH;
	wCom = zeroC;

	// cell weights
	wIc.assign(nCells, zeroI);
	wIcd.assign(nCells, zeroI);
	wIcm.assign(nCells, zeroI);

	wHc.assign(nCells, zeroH);
	wHcd.assign(nCells, zeroH);
	wHcm.assign(nCells, zeroH);

	// bias
	biasI=biasF=biasO=biasId=biasFd=biasOd=biasIm=biasFm=biasOm=0.;
	biasC.assign(nCells, 0.);
	biasCd.assign(nCells, 0.);
	biasCm.assign(nCells, 0.);
}

void LstmBlock::setConstantWeights(prec w){
	wIi.assign(nInputs, w);
	wIf.assign(nInputs, w);
	wIo.assign(nInputs, w);
	wHi.assign(nBxnC, w);
	wHf.assign(nBxnC, w);
	wHo.assign(nBxnC, w);
	wCi.assign(nCells, w);
	wCf.assign(nCells, w);
	wCo.assign(nCells, w);
	for(int i=0;i<nCells;i++){
		wIc[i].assign(nInputs, w);
		wHc[i].assign(nBxnC, w);
	}
	biasC.assign(nCells, w);	
	biasI=w;
	biasF=w;
	biasO=w;
}

void LstmBlock::setRandomWeights(prec halfRange){

	for(int i=0;i<nInputs;i++){
		wIi[i]=randomW(halfRange);
		wIf[i]=randomW(halfRange);
		wIo[i]=randomW(halfRange);
	}
	for(int i=0;i<nBxnC;i++){
		wHi[i]=randomW(halfRange);
		wHf[i]=randomW(halfRange);
		wHo[i]=randomW(halfRange);
	}
	for(int i=0;i<nCells;i++){
		wCi[i]=randomW(halfRange);
		wCf[i]=randomW(halfRange);
		wCo[i]=randomW(halfRange);
	}
	for(int i=0;i<nCells;i++){
		for(int j=0;j<nInputs;j++){
			wIc[i][j] = randomW(halfRange);
		}
		for(int j=0;j<nBxnC;j++){
			wHc[i][j] = randomW(halfRange);
		}
		biasC[i] = randomW(halfRange);
	}

	biasI=randomW(halfRange);
	biasF=randomW(halfRange);
	biasO=randomW(halfRange);
}

// computes single forward pass step of an LSTM memory block, operates on a vector of state structs
void LstmBlock::forwardPassStep(LstmBlockState& thisState, LstmBlockState& prevState, std::vector<prec>& input, bool skipPrev, std::vector<prec>& pbcs){
	
	// input gate
	thisState.ai = inner_product(&wIi[0], &input[0], nInputs);

	if(!skipPrev) {
		thisState.ai += inner_product(&wHi[0], &pbcs[0], nBxnC);
		thisState.ai += inner_product(&wCi[0], &prevState.sc[0], nCells);
	}
	thisState.ai += biasI; // add bias
	thisState.bi = fnF(thisState.ai);

	// forget gate
	thisState.af = inner_product(&wIf[0], &input[0], nInputs);
	if(!skipPrev) {
		thisState.af += inner_product(&wHf[0], &pbcs[0], nBxnC);
		thisState.af += inner_product(&wCf[0], &prevState.sc[0], nCells);
	}
	thisState.af += biasF; // bias
	thisState.bf = fnF(thisState.af);

	// cells
	if (skipPrev) {
		for(int i=0;i<nCells;i++) {
			thisState.ac[i] = inner_product(&wIc[i][0], &input[0], nInputs)
			               + biasC[i]; // bias
			thisState.sc[i] = thisState.bi * fnG(thisState.ac[i]); // new cell state
		}
	} else {
		for(int i=0;i<nCells;i++) { // cells
			thisState.ac[i] = inner_product(&wIc[i][0], &input[0], nInputs)
				           + inner_product(&wHc[i][0], &pbcs[0], nBxnC) // mem. block input activation
			               + biasC[i]; // bias
			thisState.sc[i] = thisState.bi * fnG(thisState.ac[i]) // new cell state
						   + thisState.bf * prevState.sc[i];
		}
	}

	// output gate
	thisState.ao = inner_product(&wIo[0], &input[0], nInputs)
		        + inner_product(&wCo[0], &thisState.sc[0], nCells); // output gate activation
	if(!skipPrev){
		thisState.ao += inner_product(&wHo[0], &pbcs[0], nBxnC);
	}
	
	thisState.ao += biasO; // add bias
	thisState.bo = fnF(thisState.ao);
	for(int i=0;i<nCells;i++){
		thisState.bc[i] = thisState.bo * fnH(thisState.sc[i]); // cell outputs (memory block block output)
	}
}

// the wK matrix is a corresponding columns from wK matrix stored in ForwardLayer organized in a matrix by rows
// the wH matrix is a part of transposed matrix (columns as rows) for the source cells (recurrent connections)
// dk is a vector of deltas from output layer, db is vector of deltas of all cells (not squashed) in time t+1
void LstmBlock::backwardPassStep(LstmBlockState& st, LstmBlockState& prevst, LstmBlockState& nextst, 
		std::vector< std::vector<prec> >& wK, std::vector<prec>& dk, 
		std::vector< std::vector<prec> >& wH, std::vector<prec>& db,
		std::vector< std::vector<prec> >& gw, std::vector<prec>& gd, 
		std::vector<prec>& input, bool skipPrev, bool skipNext, 
		std::vector<prec>& pbcs){
	

	// output gate
	st.d_o = 0.; // epochal bugfix :)
	if (skipNext){
		for(int i=0; i<nCells; i++){
			st.ec[i] = inner_product(&wK[i][0], &dk[0], nOutputs) // error propagated from outputs
		                   + inner_product(&wH[i][0], &db[0], nBxnC);   // error propagated through recurrent connections from the cells
			st.d_o += fnH(st.sc[i]) * st.ec[i]; // accumulate errors in state
		}
		st.d_o *= fnFd(st.ao); // output gate delta
		for(int i=0; i<nCells; i++){
			st.es[i] = st.bo * fnHd(st.sc[i]) * st.ec[i] + wCo[i] * st.d_o;
			st.dc[i] = st.bi * fnGd(st.ac[i]) * st.es[i];
		}
	} else {
		for(int i=0; i<nCells; i++){
			st.ec[i] = inner_product(&wK[i][0], &dk[0], nOutputs) // error propagated from outputs
		                   + inner_product(&wH[i][0], &db[0], nBxnC)   // error propagated through recurrent connections from the cells
						   + inner_product(&gw[i][0], &gd[0], nBlocks*3); // this is missing in Alex's equations (4.12)
			st.d_o += fnH(st.sc[i]) * st.ec[i]; // accumulate errors in state
		}
		st.d_o *= fnFd(st.ao); // output gate delta
		for(int i=0; i<nCells; i++){
			st.es[i] = st.bo * fnHd(st.sc[i]) * st.ec[i] + wCo[i] * st.d_o;
			st.es[i] += nextst.bf * nextst.es[i] + wCi[i] * nextst.di + wCf[i] * nextst.df;
			st.dc[i] = st.bi * fnGd(st.ac[i]) * st.es[i];
		}
	}

	// forget gate
	if(skipPrev){
		st.df = 0.;
	} else {
		st.df = fnFd(st.af) * inner_product(&prevst.sc[0], &st.es[0], nCells);
	}

	// input gate
	st.di = 0.;
	for(int i=0; i<nCells; i++){
		st.di += fnG(st.ac[i]) * st.es[i]; // accumulate  errors for input gate
	}
	st.di *= fnFd(st.ai);

	// update weight derivatives
	vector_add_scale(&wIid[0], &input[0], st.di, nInputs);
	vector_add_scale(&wIfd[0], &input[0], st.df, nInputs);
	vector_add_scale(&wIod[0], &input[0], st.d_o, nInputs);


	for(int j=0; j<nCells; j++){ // reordered loop for speed
		vector_add_scale( &wIcd[j][0], &input[0], st.dc[j], nInputs);
	}

	if(!skipPrev) {
		vector_add_scale(&wHid[0], &pbcs[0], st.di, nBxnC);
		vector_add_scale(&wHfd[0], &pbcs[0], st.df, nBxnC);
		vector_add_scale(&wHod[0], &pbcs[0], st.d_o, nBxnC);

		for(int j=0; j<nCells; j++){ // reordered loop for speed
			vector_add_scale(&wHcd[j][0], &pbcs[0], st.dc[j], nBxnC);
		}
	}

	if(!skipPrev){
		vector_add_scale(&wCid[0], &prevst.sc[0], st.di, nCells);
		vector_add_scale(&wCfd[0], &prevst.sc[0], st.df, nCells);
	}
	vector_add_scale(&wCod[0], &st.sc[0], st.d_o, nCells); //TODO: ask Jan: shouldn't this also be state[t-1]?

	// update bias derivatives
	biasId += st.di;
	biasFd += st.df;
	biasOd += st.d_o;

	vector_add(&biasCd[0], &st.dc[0], nCells);
}


void LstmBlock::updateWeights(prec eta, prec alpha){
	// weights from inputs
	scalar_multiple(eta,   &wIid[0], nInputs);
	scalar_multiple(alpha, &wIim[0], nInputs);
	vector_add(&wIim[0], &wIid[0], nInputs);
	vector_add(&wIi[0],  &wIim[0], nInputs);

	scalar_multiple(eta,   &wIfd[0], nInputs);
	scalar_multiple(alpha, &wIfm[0], nInputs);
	vector_add(&wIfm[0], &wIfd[0], nInputs);
	vector_add(&wIf[0],  &wIfm[0], nInputs);

	scalar_multiple(eta,   &wIod[0], nInputs);
	scalar_multiple(alpha, &wIom[0], nInputs);
	vector_add(&wIom[0], &wIod[0], nInputs);
	vector_add(&wIo[0],  &wIom[0], nInputs);

	// peephole weights
	scalar_multiple(eta,   &wCid[0], nCells);
	scalar_multiple(alpha, &wCim[0], nCells);
	vector_add(&wCim[0], &wCid[0], nCells);
	vector_add(&wCi[0],  &wCim[0], nCells);

	scalar_multiple(eta,   &wCfd[0], nCells);
	scalar_multiple(alpha, &wCfm[0], nCells);
	vector_add(&wCfm[0], &wCfd[0], nCells);
	vector_add(&wCf[0],  &wCfm[0], nCells);

	scalar_multiple(eta,   &wCod[0], nCells);
	scalar_multiple(alpha, &wCom[0], nCells);
	vector_add(&wCom[0], &wCod[0], nCells);
	vector_add(&wCo[0],  &wCom[0], nCells);

	// weights from other cells (recurrent connections)
	scalar_multiple(eta,   &wHid[0], nBxnC);
	scalar_multiple(alpha, &wHim[0], nBxnC);
	vector_add(&wHim[0], &wHid[0], nBxnC);
	vector_add(&wHi[0],  &wHim[0], nBxnC);

	scalar_multiple(eta,   &wHfd[0], nBxnC);
	scalar_multiple(alpha, &wHfm[0], nBxnC);
	vector_add(&wHfm[0], &wHfd[0], nBxnC);
	vector_add(&wHf[0],  &wHfm[0], nBxnC);

	scalar_multiple(eta,   &wHod[0], nBxnC);
	scalar_multiple(alpha, &wHom[0], nBxnC);
	vector_add(&wHom[0], &wHod[0], nBxnC);
	vector_add(&wHo[0],  &wHom[0], nBxnC);

	// cells weights
	for(int i=0; i<nCells; i++){ // for
		scalar_multiple(eta,   &wIcd[i][0], nInputs);
		scalar_multiple(alpha, &wIcm[i][0], nInputs);
		vector_add(&wIcm[i][0], &wIcd[i][0], nInputs);
		vector_add(&wIc[i][0],  &wIcm[i][0], nInputs);

		scalar_multiple(eta,   &wHcd[i][0], nBxnC);
		scalar_multiple(alpha, &wHcm[i][0], nBxnC);
		vector_add(&wHcm[i][0], &wHcd[i][0], nBxnC);
		vector_add(&wHc[i][0],  &wHcm[i][0], nBxnC);
	}

	// bias
	biasIm = eta * biasId + alpha * biasIm;
	biasI += biasIm;
	biasFm = eta * biasFd + alpha * biasFm;
	biasF += biasFm;
	biasOm = eta * biasOd + alpha * biasOm;
	biasO += biasOm;

	scalar_multiple(eta,   &biasCd[0], nCells);
	scalar_multiple(alpha, &biasCm[0], nCells);
	vector_add(&biasCm[0], &biasCd[0], nCells); // learning
	vector_add(&biasC[0],  &biasCm[0], nCells); // momentum
}

void LstmBlock::resetDerivs(){
	std::vector<prec> zeroI(nInputs, 0.);
	std::vector<prec> zeroC(nCells, 0.);
	std::vector<prec> zeroH(nBxnC, 0.);

	wIid = zeroI;
	wHid = zeroH;
	wCid = zeroC;

	wIfd = zeroI;
	wHfd = zeroH;
	wCfd = zeroC;

	wIod = zeroI;
	wHod = zeroH;
	wCod = zeroC;

	wIcd.assign(nCells, zeroI);
	wHcd.assign(nCells, zeroH);

	biasId=biasFd=biasOd=0.;
	biasCd = zeroC;
}

void LstmBlock::getSerializedWeights1(std::vector<prec> & w){
	//w.reserve(w.size() + nInputs + nInputs + (nCells * nInputs) + nInputs + nCells + nCells + nCells);
	
	w.insert(w.end(), wIi.begin(), wIi.end());
	w.insert(w.end(), wIf.begin(), wIf.end());
	for(int i=0;i<nCells;i++){
		w.insert(w.end(), wIc[i].begin(), wIc[i].end());
	}
	w.insert(w.end(), wIo.begin(), wIo.end());
	w.insert(w.end(), wCi.begin(), wCi.end());
	w.insert(w.end(), wCf.begin(), wCf.end());
	w.insert(w.end(), wCo.begin(), wCo.end());
}

void LstmBlock::setSerializedWeights1(std::vector<prec>::iterator & it){
	wIi.assign(it, it+nInputs); it += nInputs;
	wIf.assign(it, it+nInputs); it += nInputs;
	for(int i=0;i<nCells;i++){
		wIc[i].assign(it, it+nInputs); it += nInputs;
	}
	wIo.assign(it, it+nInputs); it += nInputs;
	wCi.assign(it, it+nCells); it += nCells;
	wCf.assign(it, it+nCells); it += nCells;
	wCo.assign(it, it+nCells); it += nCells;
}

void LstmBlock::getSerializedWeights2(std::vector<prec> & w){
	//w.reserve(w.size() + nBxnC + nBxnC + (nCells * nBxnC) + nBxnC + 3 + nCells);
	
	w.insert(w.end(), wHi.begin(), wHi.end());
	w.insert(w.end(), wHf.begin(), wHf.end());
	for(int i=0;i<nCells;i++){
		w.insert(w.end(), wHc[i].begin(), wHc[i].end());
	}
	w.insert(w.end(), wHo.begin(), wHo.end());
	w.push_back(biasI);
	w.push_back(biasF);
	w.push_back(biasO);
	w.insert(w.end(), biasC.begin(), biasC.end());
}

void LstmBlock::setSerializedWeights2(std::vector<prec>::iterator & it){
	wHi.assign(it, it+nBxnC); it += nBxnC;
	wHf.assign(it, it+nBxnC); it += nBxnC;
	for(int i=0;i<nCells;i++){
		wHc[i].assign(it, it+nBxnC); it += nBxnC;
	}
	wHo.assign(it, it+nBxnC); it += nBxnC;
	biasI = *(it++);
	biasF = *(it++);
	biasO = *(it++);
	biasC.assign(it, it+nCells); it += nCells;
}

void LstmBlock::getSerializedDerivs1(std::vector<prec> & w){
	//w.reserve(w.size() + (6 * nInputs) + (6 * nCells) + (2 * nCells * nInputs));

	// derivatives
	w.insert(w.end(), wIid.begin(), wIid.end());
	w.insert(w.end(), wIfd.begin(), wIfd.end());
	for(int i=0;i<nCells;i++){
		w.insert(w.end(), wIcd[i].begin(), wIcd[i].end());
	}
	w.insert(w.end(), wIod.begin(), wIod.end());
	w.insert(w.end(), wCid.begin(), wCid.end());
	w.insert(w.end(), wCfd.begin(), wCfd.end());
	w.insert(w.end(), wCod.begin(), wCod.end());

	// momentum
	w.insert(w.end(), wIim.begin(), wIim.end());
	w.insert(w.end(), wIfm.begin(), wIfm.end());
	for(int i=0;i<nCells;i++){
		w.insert(w.end(), wIcm[i].begin(), wIcm[i].end());
	}
	w.insert(w.end(), wIom.begin(), wIom.end());
	w.insert(w.end(), wCim.begin(), wCim.end());
	w.insert(w.end(), wCfm.begin(), wCfm.end());
	w.insert(w.end(), wCom.begin(), wCom.end());
}

void LstmBlock::getSerializedDerivs2(std::vector<prec> & w){
	//w.reserve(w.size() + (6 * nBxnC) + (2 * nCells * nBxnC) + 6 + (2 * nCells));
	
	// derivatives
	w.insert(w.end(), wHid.begin(), wHid.end());
	w.insert(w.end(), wHfd.begin(), wHfd.end());
	for(int i=0;i<nCells;i++){
		w.insert(w.end(), wHcd[i].begin(), wHcd[i].end());
	}
	w.insert(w.end(), wHod.begin(), wHod.end());
	w.push_back(biasId);
	w.push_back(biasFd);
	w.push_back(biasOd);
	w.insert(w.end(), biasCd.begin(), biasCd.end());

	// derivatives
	w.insert(w.end(), wHim.begin(), wHim.end());
	w.insert(w.end(), wHfm.begin(), wHfm.end());
	for(int i=0;i<nCells;i++){
		w.insert(w.end(), wHcm[i].begin(), wHcm[i].end());
	}
	w.insert(w.end(), wHom.begin(), wHom.end());
	w.push_back(biasIm);
	w.push_back(biasFm);
	w.push_back(biasOm);
	w.insert(w.end(), biasCm.begin(), biasCm.end());
}

void LstmBlock::setSerializedDerivs1(std::vector<prec>::iterator & it){
	
	// derivatives
	wIid.assign(it, it+nInputs); it += nInputs;
	wIfd.assign(it, it+nInputs); it += nInputs;
	for(int i=0;i<nCells;i++){
		wIcd[i].assign(it, it+nInputs); it += nInputs;
	}
	wIod.assign(it, it+nInputs); it += nInputs;
	wCid.assign(it, it+nCells); it += nCells;
	wCfd.assign(it, it+nCells); it += nCells;
	wCod.assign(it, it+nCells); it += nCells;

	// momentum
	wIim.assign(it, it+nInputs); it += nInputs;
	wIfm.assign(it, it+nInputs); it += nInputs;
	for(int i=0;i<nCells;i++){
		wIcm[i].assign(it, it+nInputs); it += nInputs;
	}
	wIom.assign(it, it+nInputs); it += nInputs;
	wCim.assign(it, it+nCells); it += nCells;
	wCfm.assign(it, it+nCells); it += nCells;
	wCom.assign(it, it+nCells); it += nCells;
}

void LstmBlock::setSerializedDerivs2(std::vector<prec>::iterator & it){
	
	// derivatives
	wHid.assign(it, it+nBxnC); it += nBxnC;
	wHfd.assign(it, it+nBxnC); it += nBxnC;
	for(int i=0;i<nCells;i++){
		wHcd[i].assign(it, it+nBxnC); it += nBxnC;
	}
	wHod.assign(it, it+nBxnC); it += nBxnC;
	biasId = *(it++);
	biasFd = *(it++);
	biasOd = *(it++);
	biasCd.assign(it, it+nCells); it += nCells;

	// momentum
	wHim.assign(it, it+nBxnC); it += nBxnC;
	wHfm.assign(it, it+nBxnC); it += nBxnC;
	for(int i=0;i<nCells;i++){
		wHcm[i].assign(it, it+nBxnC); it += nBxnC;
	}
	wHom.assign(it, it+nBxnC); it += nBxnC;
	biasIm = *(it++);
	biasFm = *(it++);
	biasOm = *(it++);
	biasCm.assign(it, it+nCells); it += nCells;
}

LstmBlock::~LstmBlock() {

}
