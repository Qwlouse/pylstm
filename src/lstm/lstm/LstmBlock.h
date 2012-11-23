/*
 * LstmBlock.h
 *
 *  Created on: Oct 26, 2010
 *      Author: koutnij
 */
#include <vector>
#include <math.h>

#include "Constants.h"

#ifndef LSTMBLOCK_H_
#define LSTMBLOCK_H_

struct LstmBlockState{
	LstmBlockState(int nCells=1){ // default constructor creates one cell inside
		this->nCells = nCells;
		std::vector<prec> zeroC (nCells, 0.); // "=" is copy operator for vector
		sc = zeroC;
		ac = zeroC;
		bc = zeroC;
		ec = zeroC;
		es = zeroC;
		dc = zeroC;
		ai=af=ao=bi=bf=bo=di=df=d_o=0.0; // this is necessary, C++ does not initialize them in vector of states
	}
	void addCell(){
		nCells++;
		sc.push_back(0.);
		ac.push_back(0.);
		bc.push_back(0.);
		ec.push_back(0.);
		es.push_back(0.);
		dc.push_back(0.);
	}
	void setSerializedBlockStates(std::vector<prec>::iterator & it){
		sc.assign(it, it+nCells); it += nCells;
		ac.assign(it, it+nCells); it += nCells;
		bc.assign(it, it+nCells); it += nCells;
		ec.assign(it, it+nCells); it += nCells;
		es.assign(it, it+nCells); it += nCells;
		dc.assign(it, it+nCells); it += nCells;
		ai = *(it++);
		af = *(it++);
		ao = *(it++);
		bi = *(it++);
		bf = *(it++);
		bo = *(it++);
		di = *(it++);
		df = *(it++);
		d_o = *(it++);
	}
	void getSerializedBlockStates(std::vector<prec> & s){
		//s.reserve(s.size() + (6 * nCells) + 9); 
		s.insert(s.end(), sc.begin(), sc.end());
		s.insert(s.end(), ac.begin(), ac.end());
		s.insert(s.end(), bc.begin(), bc.end());
		s.insert(s.end(), ec.begin(), ec.end());
		s.insert(s.end(), es.begin(), es.end());
		s.insert(s.end(), dc.begin(), dc.end());
		s.push_back(ai);
		s.push_back(af);
		s.push_back(ao);
		s.push_back(bi);
		s.push_back(bf);
		s.push_back(bo);
		s.push_back(di);
		s.push_back(df);
		s.push_back(d_o);
	}

	int nCells;

	// current cell states
	std::vector <prec> sc; // cell states
	std::vector <prec> ac; // block input activations
	std::vector <prec> bc; // memory block output
	prec ai; // input gate excitation
	prec af; // forget gate excitation
	prec ao; // output gate excitation
	prec bi; // input gate activation
	prec bf; // forget gate activation
	prec bo; // output gate activation

	//deltas
	prec di; // input gate delta
	prec df; // forget gate delta
	prec d_o; // output gate delta
	std::vector<prec> dc; // cell delta

	//errors
	std::vector<prec> ec; // cell outputs errors
	std::vector<prec> es; // cell states errors
};


class LstmBlock {
public:
	LstmBlock(int nCells, int nInputs, int nOutputs, int nBlocks);
	void forwardPassStep(LstmBlockState& thisState, LstmBlockState& prevState, std::vector<prec>& input, bool isStartState, std::vector<prec>& pbcs);
	// the wK vector is a corresponding column from wK matrix stored in ForwardLayer
	// the wH matrix is a part of transposed matrix (columns as rows) for the source cells
	void backwardPassStep(LstmBlockState& st, LstmBlockState& prevst, LstmBlockState& nextst, 
		std::vector< std::vector<prec> >& wK, std::vector<prec>& dk, 
		std::vector< std::vector<prec> >& wH, std::vector<prec>& db,
		std::vector< std::vector<prec> >& gw, std::vector<prec>& gd, 
		std::vector<prec>& input, bool skipPrev, bool skipNext, 
		std::vector<prec>& pbcs);
	void setConstantWeights(prec w);
	void setRandomWeights(prec halfRange);
	void updateMomentum(prec alpha);
	void updateWeights(prec eta, prec alpha);
	void resetDerivs();

	void getSerializedWeights1(std::vector<prec> & w);
	void setSerializedWeights1(std::vector<prec>::iterator & it);
	void getSerializedWeights2(std::vector<prec> & w);
	void setSerializedWeights2(std::vector<prec>::iterator & it);
	void getSerializedDerivs1(std::vector<prec> & w);
	void setSerializedDerivs1(std::vector<prec>::iterator & it);
	void getSerializedDerivs2(std::vector<prec> & w);
	void setSerializedDerivs2(std::vector<prec>::iterator & it);
	
	~LstmBlock();


//private:
	int nCells, nInputs, nOutputs, nBlocks, nBxnC;

	//LstmBlockState state;

	// input gate weights, derivatives, momentum weights
	std::vector<prec> wIi;
	std::vector<prec> wHi;
	std::vector<prec> wCi;
	std::vector<prec> wIid;
	std::vector<prec> wHid;
	std::vector<prec> wCid;
	std::vector<prec> wIim;
	std::vector<prec> wHim;
	std::vector<prec> wCim;

    // forget gate weights
	std::vector<prec> wIf;
	std::vector<prec> wHf;
	std::vector<prec> wCf;
	std::vector<prec> wIfd;
	std::vector<prec> wHfd;
	std::vector<prec> wCfd;
	std::vector<prec> wIfm;
	std::vector<prec> wHfm;
	std::vector<prec> wCfm;

	// output gate weights
	std::vector<prec> wIo;
	std::vector<prec> wHo;
	std::vector<prec> wCo;
	std::vector<prec> wIod;
	std::vector<prec> wHod;
	std::vector<prec> wCod;
	std::vector<prec> wIom;
	std::vector<prec> wHom;
	std::vector<prec> wCom;

	// cell weights
	std::vector< std::vector<prec> > wIc;
	std::vector< std::vector<prec> > wHc;
	std::vector< std::vector<prec> > wIcd;
	std::vector< std::vector<prec> > wHcd;
	std::vector< std::vector<prec> > wIcm;
	std::vector< std::vector<prec> > wHcm;

	// bias
	prec biasI, biasF, biasO, biasId, biasFd, biasOd, biasIm, biasFm, biasOm;
	std::vector <prec> biasC;
	std::vector <prec> biasCd;
	std::vector <prec> biasCm;
};


inline prec fnF(prec x){
	return 1/(1+exp(-x));
}
inline prec fnFd(prec x){
	prec ex=exp(x);
	return ex/((1+ex)*(1+ex));
}

inline prec fnG(prec x){
	//return 1/(1+exp(-x));
	return tanh(x);
}
inline prec fnGd(prec x){
	//prec ex=exp(x);
	//return ex/((1+ex)*(1+ex));
	prec th = tanh(x);
	return 1 - th * th;
}

inline prec fnH(prec x){
	//return 1/(1+exp(-x));
	return tanh(x);
}
inline prec fnHd(prec x){
	//prec ex=exp(x);
	//return ex/((1+ex)*(1+ex));
	prec th = tanh(x);
	return 1 - th * th;
}

#endif /* LSTMBLOCK_H_ */
