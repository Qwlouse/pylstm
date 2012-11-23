/*
 * ForwardLayer.h
 *
 *  Created on: Oct 27, 2010
 *      Author: koutnij
 */

#include <vector>
#include <math.h>

#include "Constants.h"

#ifndef FORWARDLAYER_H_
#define FORWARDLAYER_H_

struct ForwardLayerState{
	ForwardLayerState() {};
	ForwardLayerState(int nOutputs){
		this->nOutputs = nOutputs;
		std::vector<prec> zeroO(nOutputs, 0.);
		ak = zeroO;
		bk = zeroO;
		ek = zeroO;
		dk = zeroO;
	}
	void setSerializedStates(std::vector<prec>::iterator & it){
		ak.assign(it, it+nOutputs); it += nOutputs;
		bk.assign(it, it+nOutputs); it += nOutputs;
		ek.assign(it, it+nOutputs); it += nOutputs;
		dk.assign(it, it+nOutputs); it += nOutputs;
	}
	void getSerializedStates(std::vector<prec> & s){
		//s.reserve(s.size() + (4 * nOutputs)); 
		s.insert(s.end(), ak.begin(), ak.end());
		s.insert(s.end(), bk.begin(), bk.end());
		s.insert(s.end(), ek.begin(), ek.end());
		s.insert(s.end(), dk.begin(), dk.end());
	}

	int nOutputs;

	// current cell states
	std::vector<prec> ak; // neuron excitations
	std::vector<prec> bk; // neuron activations
	std::vector<prec> ek; // errors
	std::vector<prec> dk; // deltas
};

class ForwardLayer {
public:
	ForwardLayer(int nInputs, int nOutputs);
	void forwardPassStep(ForwardLayerState& thisState, std::vector<prec>& input);
	void backwardPassStep(ForwardLayerState& thisState, std::vector<prec>& input, std::vector<prec>& er);
	void setConstantWeights(prec w);
	void setRandomWeights(prec halfRange);
	void updateWeights(prec eta, prec alpha);
	void resetDerivs();

	void getSerializedWeights(std::vector<prec> & w);
	void setSerializedWeights(std::vector<prec>::iterator & it);
	void getSerializedDerivs(std::vector<prec> & w);
	void setSerializedDerivs(std::vector<prec>::iterator & it);

	~ForwardLayer();

	int nInputs, nOutputs;
	std::vector< std::vector<prec> > wK;
	std::vector< std::vector<prec> > wKd; // weight derivatives, updated through the backward pass
	std::vector< std::vector<prec> > wKm;	// weights after the previous training episode - for momentum

	std::vector <prec> biasK; // bias vector, biases are treated separately
	std::vector <prec> biasKd;
	std::vector <prec> biasKm;
private:

};

inline prec fnO(prec x){
	//return 1/(1+exp(-x));
	return 2/(1+exp(-2*x)) - 1;
	//return x;
}

inline prec fnOd(prec x){
	//prec ex=exp(x);
	//return ex/((1+ex)*(1+ex));
	prec y = fnO(x);
	return 1 - (y*y);
	//return 1.;
}

template <class InputIterator, class OutputIterator>
   static OutputIterator fnO ( InputIterator first, InputIterator last, OutputIterator result){
   while (first!=last){
	   //*result++ = 1/(1+exp(-(*first++)));
	   *result++ = fnO((*first++));
   }
   return result;
}

template <class InputIterator, class OutputIterator>
   static OutputIterator fnOd ( InputIterator first, InputIterator last, OutputIterator result){
   prec ex;
   while (first!=last){
	    //ex = exp(*first++);
		//*result++ = ex/((1+ex)*(1+ex));
		*result++ = fnOd((*first++));
   }
   return result;
}


#endif /* FORWARDLAYER_H_ */
