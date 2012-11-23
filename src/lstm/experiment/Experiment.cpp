/*
 * Experiment.cpp
 *
 *  Created on: Nov 21, 2010
 *      Author: koutnij
 */

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <ctime>

//#include "mex.h"

#include "Experiment.h"
#include "../lstm/Constants.h"
#include "../lstm/LstmLayer.h"
#include "../lstm/ForwardLayer.h"
#include "../lstm/LstmNetwork.h"
#include "../lstm/MatlabMath.h"


Experiment::Experiment() {
}

Experiment::~Experiment() {
	delete lstmNetwork;
}

// one forward step
void Experiment::forwardStep(std::vector<prec>& serializedStates, std::vector<prec>& input, bool isStart, std::vector<prec>& output) {

	// write serializedStates
	std::vector<LstmBlockState> lstmCurrentStates(nBlocks, LstmBlockState(nCells));
	ForwardLayerState forwardCurrentState(nOutputs);
	setSerializedStates(lstmCurrentStates, forwardCurrentState, serializedStates);
	
	// wrap input and output arguments in vectors
	std::vector<std::vector<prec>> inputs(1, input);
	std::vector<bool> isStarts(1, isStart);
	std::vector<std::vector<prec>> outputs(1, output);

	// forward pass for 1 step
	lstmNetwork->forward(0, 1, lstmCurrentStates, forwardCurrentState, inputs, isStarts, outputs);

	// read states back into serializedStates
	serializedStates.clear();
	getSerializedStates(lstmCurrentStates, forwardCurrentState, serializedStates);

	// set output
	output = outputs[0];
}

// run network forward, and write final network states and outputs
void Experiment::forward(std::vector<prec>& serializedStates, 
						 std::vector<std::vector<prec>>&inputs, std::vector<bool>& isStarts, 
						 std::vector<std::vector<prec>>& outputs) {

	
	// write serializedStates
	std::vector<LstmBlockState> lstmCurrentStates(nBlocks, LstmBlockState(nCells));
	ForwardLayerState forwardCurrentState(nOutputs);
	setSerializedStates(lstmCurrentStates, forwardCurrentState, serializedStates);
	
	// forward pass
	lstmNetwork->forward(0, inputs.size(), lstmCurrentStates, forwardCurrentState, inputs, isStarts, outputs);

	// read states back into serializedStates
	serializedStates.clear();
	getSerializedStates(lstmCurrentStates, forwardCurrentState, serializedStates); // final state
}

// run network forward, store everything in the buffers, and write final network states and outputs.
void Experiment::forwardSteps(std::vector<prec>& serializedStates, std::vector<std::vector<prec> >& inputs,  
						      std::vector<bool>& isStarts, std::vector<std::vector<prec> >& outputs) {

	// set start states
	std::vector<LstmBlockState> lstmCurrentStates(nBlocks, LstmBlockState(nCells));
	ForwardLayerState forwardCurrentState(nOutputs);
	setSerializedStates(lstmCurrentStates, forwardCurrentState, serializedStates);

	// create buffers and outputs
	if (buffersLength != inputs.size()) {
		buffersLength = inputs.size();
		std::vector<LstmBlockState> lstmBlockStates(nBlocks, LstmBlockState(nCells));
		lstmStateBuffer.assign(buffersLength, lstmBlockStates);
		outputLayerStateBuffer.assign(buffersLength, ForwardLayerState(nOutputs));
	}

	// forward pass
	lstmNetwork->forwardPass(0, buffersLength, lstmCurrentStates, forwardCurrentState, lstmStateBuffer, outputLayerStateBuffer, inputs, isStarts, outputs);
	
	// read states back into serializedStates
	serializedStates.clear();
	getSerializedStates(lstmCurrentStates, forwardCurrentState, serializedStates);
}

// Train, then return serializedWeights, serializedDerivs, serializedStates, and outputs.
prec Experiment::trainEpochs(std::vector<prec>& serializedWeights, std::vector<prec>& serializedDerivs,
							 std::vector<prec>& serializedStates, std::vector<std::vector<prec> >& inputs,
						     std::vector<bool>& isStarts, std::vector<std::vector<prec>>& targets, std::vector<bool>& isTargets,
							 std::vector<std::vector<prec> >& outputs, bool finalpass, prec learningRate, prec momentum, int nEpochs) {
	

	prec error = 0.;
	
	// set start states
	std::vector<LstmBlockState> lstmCurrentStates(nBlocks, LstmBlockState(nCells));
	ForwardLayerState forwardCurrentState(nOutputs);

	// create buffers, if necessary
	if (buffersLength != inputs.size()) {
		buffersLength = inputs.size();
		std::vector<LstmBlockState> lstmBlockStates(nBlocks, LstmBlockState(nCells));
		lstmStateBuffer.assign(buffersLength, lstmBlockStates);
		outputLayerStateBuffer.assign(buffersLength, ForwardLayerState(nOutputs));
		std::vector<prec> zeroOutVector(nOutputs, 0.);
		errorBuffer.assign(buffersLength, zeroOutVector);
	}

	std::vector<prec> startStates;

	// set LSTM and weights, derivatives and momentum
	lstmNetwork->setSerializedWeights(serializedWeights);
	lstmNetwork->setSerializedDerivs(serializedDerivs);

	// train for nEpochs
	for(int i=0;i<nEpochs;i++){
		lstmNetwork->resetDerivs();

		// forward pass
		setSerializedStates(lstmCurrentStates, forwardCurrentState, serializedStates); // copy start states, because they get overwritten each pass
		lstmNetwork->forwardPass(0, buffersLength, lstmCurrentStates, forwardCurrentState, lstmStateBuffer, outputLayerStateBuffer, inputs, isStarts, outputs);
		
		// compute errors where isTargets is true
		error = getOutputError(outputs, targets, isTargets, errorBuffer);

		// backward pass
		setSerializedStates(lstmCurrentStates, forwardCurrentState, serializedStates); 	// copy start states, because they get overwritten each pass
		lstmNetwork->backwardPass(0, buffersLength, lstmCurrentStates, lstmStateBuffer, outputLayerStateBuffer, errorBuffer, inputs, isStarts);

		// update weights
		lstmNetwork->updateWeights(learningRate /*/ buffersLength */, momentum);
	}

	// final forward pass after updating weights
	if (finalpass) {
		setSerializedStates(lstmCurrentStates, forwardCurrentState, serializedStates); // copy start states, because they get overwritten
		lstmNetwork->forwardPass(0, buffersLength, lstmCurrentStates, forwardCurrentState, lstmStateBuffer, outputLayerStateBuffer, inputs, isStarts, outputs);
		error = getOutputError(outputs, targets, isTargets, errorBuffer);
	}

	// get new weights, derivatives, momentum and states
	serializedWeights.clear();
	serializedDerivs.clear();
	serializedStates.clear();
		
	lstmNetwork->getSerializedWeights(serializedWeights);
	lstmNetwork->getSerializedDerivs(serializedDerivs);
	getSerializedStates(lstmCurrentStates, forwardCurrentState, serializedStates);

	// return error
	return error;
}

// set output errors in case the output is marked as a target in <isTargets>, otherwise, set to 0
prec Experiment::getOutputError(std::vector<std::vector<prec>>& outputs, 
								std::vector<std::vector<prec>>& targets, std::vector<bool>& isTargets, 
								std::vector<std::vector<prec> >& errors){
	prec error = 0.;
	int nTargetSamples = 0;
	for(size_t t=0;t<targets.size();t++){
		if (isTargets[t]) {
			//vector_difference(&state[t].bk[0], &tr[t][0], &er[t][0], nOutputs);
			vector_difference(&targets[t][0], &outputs[t][0], &errors[t][0], nOutputs);
			error += sse(&errors[t][0], nOutputs);
			nTargetSamples++;
		} else {
			errors[t].assign(nOutputs, 0.);
		}
	}

	//error =  error/(outputErrorBuffer.size() * nOutputs); // error over all samples
	error = error/((prec) 2.0 * ((prec) nTargetSamples) * nOutputs);
	return error;
}

void Experiment::setSerializedStates(std::vector<LstmBlockState>& lstmStates, ForwardLayerState& forwardState, std::vector<prec> &s) {
	if (s.size() != nStates) {
		std::cerr << "Number of states (" << s.size() << ") incorrect! Correct number is " << nStates;
		throw "number of states is incorrect!";
	}
	std::vector<prec>::iterator it = s.begin();
	for(int i=0;i<nBlocks;i++){
		lstmStates[i].setSerializedBlockStates(it);
	}
	forwardState.setSerializedStates(it);
}

void Experiment::getSerializedStates(std::vector<LstmBlockState>& lstmStates, ForwardLayerState& forwardState, std::vector<prec>& s) {
	s.reserve(nStates);
	for(int i=0;i<nBlocks;i++){
		lstmStates[i].getSerializedBlockStates(s);
	}
	forwardState.getSerializedStates(s);
}

int Experiment::getNSerializedStates(int nOutputs, int nCells, int nBlocks){
	return (nBlocks * ((6 * nCells) + 9)) + nOutputs * 4;
}

void Experiment::initLSTM(int nInputs, int nOutputs, int nCells, int nBlocks, std::vector<prec> serializedWeights) {
	this->nInputs = nInputs;
	this->nOutputs = nOutputs;
	this->nCells = nCells;
	this->nBlocks = nBlocks;
	this->nStates = getNSerializedStates(nOutputs, nCells, nBlocks);
	
	lstmStateBuffer.clear();
	outputLayerStateBuffer.clear();
	errorBuffer.clear();
	buffersLength = 0;

	// create LSTM network, and set weights
	lstmNetwork = new LstmNetwork(nInputs, nOutputs, nCells, nBlocks);
	lstmNetwork->setSerializedWeights(serializedWeights);
}

void readMatrix(std::vector<std::vector <prec> >& mat, int lineLength){
   std::ifstream indata;
   prec num;
   int cntr,lineCntr = 0;
   mat.clear();
   indata.open("data.dat");
   if(!indata){
	   std::cout << "Error reading file." << "test" << std::endl;
	   exit(1);
   }
   std::vector <prec> line;
   do{
	   cntr = 0;
	   line.clear();
	   do{
		   indata >> num;
		   line.push_back(num);
		   cntr++;
	   }while(!indata.eof() && cntr < lineLength);
	   if(line.size() == lineLength /*&& lineCntr++ < 650*/) {
		   //std::cout << "adding : ";
		   //for(int i=0;i<line.size();i++) std::cout << line[i] <<" ";
		   mat.push_back(line);
		   //std::cout << std::endl;
	   }
   }while(!indata.eof());
}


int main(){
	
	/*******************************************************************/
	/** settings *******************************************************/
	/*******************************************************************/
	std::cout.precision(16);

	//int nCells=8, nBlocks=8, nInputs=32, nX=4, nEpochs=50;
	int nCells=1, nBlocks=4, nInputs=32, nX=4;
	int nOutputs = nInputs;
	prec lr = 0.001, momentum = 0.001, nEpochs=500;
	int divd = 330; // divide data in split experiment at this sample

	int nsw = LstmNetwork::getNSerializedWeights(nInputs, nOutputs, nCells, nBlocks); // number of network weights
	int nss = Experiment::getNSerializedStates(nOutputs, nCells, nBlocks); // number of network states
	int nsd = LstmNetwork::getNSerializedDerivs(nInputs, nOutputs, nCells, nBlocks);

	std::cout << std::endl << "number of weights: " << nsw << std::endl;
	std::cout << std::endl << "number of states: " << nss << std::endl;


	/*******************************************************************/
	/** data ***********************************************************/
	/*******************************************************************/
	std::vector< std::vector<prec> > data(nX, std::vector<prec>(nInputs, 0.) );
	
	// mock data
	/*for(int i=0;i<nX;i++){
		for(int j=0;j<nInputs;j++){
			if(j == i%(nX-1)) data[i][j] = 1.; else data[i][j] = 0.;
		}
	}*/

	// load data from file
	data.clear();
	readMatrix(data, 32);

	std::vector<prec> zeroOutVector(nOutputs, 0.);
	std::vector< std::vector<prec> > inputs(data.begin(), data.end()-1);
	std::vector< std::vector<prec> > targets(data.begin() + 1, data.end());
	std::vector< std::vector<prec> > inputsP1(inputs.begin(), inputs.begin() + divd);
	std::vector< std::vector<prec> > inputsP2(inputs.begin() + divd, inputs.end());
	std::vector< std::vector<prec> > outputsTrain(inputs.size(), zeroOutVector);
	std::vector< std::vector<prec> > outputsJan(inputs.size(), zeroOutVector);
	std::vector< std::vector<prec> > outputsSteps(inputs.size(), zeroOutVector);
	std::vector< std::vector<prec> > outputsP1(outputsTrain.begin(), outputsTrain.begin() + divd);
	std::vector< std::vector<prec> > outputsP2(outputsTrain.begin() + divd, outputsTrain.end());
	std::vector< std::vector<prec> > outputsP3(outputsTrain.begin(), outputsTrain.begin() + divd);
	std::vector< std::vector<prec> > outputsP4(outputsTrain.begin() + divd, outputsTrain.end());

	std::vector<bool> isStarts(data.size() - 1, false);
	//isStarts[0] = true;
	std::vector<bool> isStartsP1(isStarts.begin(), isStarts.begin() + divd);
	std::vector<bool> isStartsP2(isStarts.begin() + divd, isStarts.end());
	std::vector<bool> isTargets(data.size() - 1, true);
	std::vector<bool> isTargetsP1(isTargets.begin(), isTargets.begin() + divd);
	std::vector<bool> isTargetsP2(isTargets.begin() + divd, isTargets.end());

	std::vector<prec> currentStates(nss, 0.);
	std::vector<prec> stepStates(nss, 0.);
	
	std::vector<prec> initialWeights(nsw, 0.);
	std::vector<prec> initialDerivs(nsd, 0.);
	std::vector<prec> trainedWeightsJan;
	std::vector<prec> trainedDerivsJan;
	std::vector<prec> trainedWeightsTrain;
	std::vector<prec> trainedDerivsTrain;


	/*******************************************************************/
	/** Jan's old functions: *******************************************/
	/*******************************************************************/
	Experiment* experJan = new Experiment();
	//experJan->initializeOneStepPrediction(nCells, nBlocks, data, janInputs, janTargets, janIsTargets, janIsStarts);
	experJan->initLSTM(nInputs, nOutputs, nCells, nBlocks, initialWeights);
	
	// randomize weights
	//srand(time(0));
	//experJan->lstmNetwork->setRandomWeights(0.005);
	experJan->lstmNetwork->setConstantWeights(0.005);
	initialWeights.clear();
	initialDerivs.clear();
	experJan->lstmNetwork->getSerializedWeights(initialWeights);
	experJan->lstmNetwork->getSerializedDerivs(initialDerivs);

	// train
	trainedWeightsJan = initialWeights;
	trainedDerivsJan = initialDerivs;
	for(int i=0;i<nEpochs;i++){
		//std::cout << i <<" MSE = "<< experJan->trainingEpoch(lr, momentum) << std::endl;
		currentStates.assign(currentStates.size(), 0.);
		std::cout << i <<" MSE = "<< experJan->trainEpochs(trainedWeightsJan, trainedDerivsJan, currentStates, inputs, isStarts, targets, isTargets, outputsJan, true, lr, momentum, 1) << std::endl;
	}

	
	/*******************************************************************/
	/** step forward steps *********************************************/
	/*******************************************************************/
	Experiment* experTrain = new Experiment();
	trainedWeightsTrain = initialWeights;
	trainedDerivsTrain  = initialDerivs;

	experTrain->initLSTM(nInputs, nOutputs, nCells, nBlocks, initialWeights);
	currentStates.assign(currentStates.size(), 0.);
	experTrain->trainEpochs(trainedWeightsTrain, trainedDerivsTrain, currentStates, inputs, isStarts, targets, isTargets, outputsTrain, true, lr, momentum, nEpochs);
	
	Experiment* experStep = new Experiment();
	experStep->initLSTM(nInputs, nOutputs, nCells, nBlocks, trainedWeightsTrain);

	// run step-wise
	for (size_t i = 0; i<data.size() - 1; i++) {
		experStep->forwardStep(stepStates, data[i], isStarts[i], outputsSteps[i]);
	}

	
	/*******************************************************************/
	/** split forward steps ********************************************/
	/*******************************************************************/
	Experiment* experSplit1 = new Experiment();
	Experiment* experSplit2 = new Experiment();
	experSplit1->initLSTM(nInputs, nOutputs, nCells, nBlocks, trainedWeightsTrain);
	currentStates.assign(currentStates.size(), 0.);
	experSplit1->forwardSteps(currentStates, inputsP1, isStartsP1, outputsP1);
	experSplit2->initLSTM(nInputs, nOutputs, nCells, nBlocks, trainedWeightsTrain);
	experSplit2->forwardSteps(currentStates, inputsP2, isStartsP2, outputsP2);
	std::vector< std::vector<prec> > outputSplit12(outputsP1.begin(), outputsP1.end());
	outputSplit12.insert(outputSplit12.end(), outputsP2.begin(), outputsP2.end()); // concatenate outputs


	/*******************************************************************/
	/** split forward **************************************************/
	/*******************************************************************/
	Experiment* experSplit3 = new Experiment();
	Experiment* experSplit4 = new Experiment();
	experSplit3->initLSTM(nInputs, nOutputs, nCells, nBlocks, trainedWeightsTrain);
	currentStates.assign(currentStates.size(), 0.);
	experSplit3->forward(currentStates, inputsP1, isStartsP1, outputsP3);
	experSplit4->initLSTM(nInputs, nOutputs, nCells, nBlocks, trainedWeightsTrain);
	experSplit4->forward(currentStates, inputsP2, isStartsP2, outputsP4);
	std::vector< std::vector<prec> > outputSplit34(outputsP3.begin(), outputsP3.end());
	outputSplit34.insert(outputSplit34.end(), outputsP4.begin(), outputsP4.end()); // concatenate outputs


	/*******************************************************************/
	/** compute differences between the methods ************************/
	/*******************************************************************/
	std::cout << "check second train (should be all zeros):" << std::endl;
	std::vector<prec> df(nOutputs); // difference
	prec sd;
	for (size_t i = 0; i<outputsJan.size(); i++) {
		vector_difference(&outputsJan[i][0], &outputsTrain[i][0], &df[0], nOutputs);
		sd = sse(&df[0], nOutputs);
		if (sd < 1e-31) {sd = 0.;};
		std::cout << sd << " ";
	}
	std::cout << std::endl << std::endl;

	std::cout << "check step-wise (should be all zeros):" << std::endl;
	for (size_t i = 0; i<outputsJan.size(); i++) {
		vector_difference(&outputsJan[i][0], &outputsSteps[i][0], &df[0], nOutputs);
		sd = sse(&df[0], nOutputs);
		if (sd < 1e-31) {sd = 0.;};
		std::cout << sd << " ";
	}
	std::cout << std::endl << std::endl;

	// difference between restart and single pass
	std::cout << "check split full pass (should be all zeros):" << std::endl;
	for (size_t i = 0; i<outputsJan.size(); i++) {
		vector_difference(&outputsJan[i][0], &outputSplit12[i][0], &df[0], nOutputs);
		sd = sse(&df[0], nOutputs);
		if (sd < 1e-31) {sd = 0.;};
		std::cout << sd << " ";
	}
	std::cout << std::endl << std::endl;


	// difference between restart and single pass
	std::cout << "check split forward only (should be all zeros):" << std::endl;
	for (size_t i = 0; i<outputsJan.size(); i++) {
		vector_difference(&outputsJan[i][0], &outputSplit34[i][0], &df[0], nOutputs);
		sd = sse(&df[0], nOutputs);
		if (sd < 1e-31) {sd = 0.;};
		std::cout << sd << " ";
	}
	std::cout << std::endl << std::endl;


	/*******************************************************************/
	/** check get and set methods **************************************/
	/*******************************************************************/
	Experiment* expercheck = new Experiment();
	expercheck->initLSTM(nInputs, nOutputs, nCells, nBlocks, trainedWeightsTrain);

	std::vector<prec> checkStatesIn, checkStatesOut, checkStatesDf;
	std::vector<prec> checkWeightsIn, checkWeightsOut, checkWeightsDf;
	std::vector<prec> checkDerivsIn, checkDerivsOut, checkDerivsDf;

	checkStatesIn.assign(nss, 0.); checkStatesDf.assign(nss, 1.);
	checkWeightsIn.assign(nsw, 0.); checkWeightsDf.assign(nsw, 1.);
	checkDerivsIn.assign(nsd, 0.); checkDerivsDf.assign(nsd, 1.);

	for (int i=0; i<nss; i++) {checkStatesIn[i]  = (prec) i;};
	for (int i=0; i<nsw; i++) {checkWeightsIn[i] = (prec) i;};
	for (int i=0; i<nsd; i++) {checkDerivsIn[i]  = (prec) i;};
	
	
	// setter
	std::vector<LstmBlockState> setLstmStates(nBlocks, LstmBlockState(nCells));
	ForwardLayerState setForwardState(nOutputs);
	expercheck->setSerializedStates(setLstmStates, setForwardState, checkStatesIn);
	expercheck->lstmNetwork->setSerializedWeights(checkWeightsIn);
	expercheck->lstmNetwork->setSerializedDerivs(checkDerivsIn);

	// getter
	expercheck->getSerializedStates(setLstmStates, setForwardState, checkStatesOut);
	expercheck->lstmNetwork->getSerializedWeights(checkWeightsOut);
	expercheck->lstmNetwork->getSerializedDerivs(checkDerivsOut);

	// difference
	vector_difference(&checkWeightsIn[0], &checkWeightsOut[0], &checkWeightsDf[0], nsw);
	vector_difference(&checkStatesIn[0], &checkStatesOut[0], &checkStatesDf[0], nss);
	vector_difference(&checkDerivsIn[0], &checkDerivsOut[0], &checkDerivsDf[0], nsd);
	
	std::cout << "weights total " << sse(&checkWeightsOut[0], nsw) << std::endl
		      << "set->get weights difference: " << sse(&checkWeightsDf[0], nsw)
			  << std::endl << std::endl;
	
	std::cout << "states total " << sse(&checkStatesOut[0], nss) << std::endl
		      << "set->get states difference: " << sse(&checkStatesDf[0], nss)
			  << std::endl << std::endl;
	
	std::cout << "derivs total " << sse(&checkDerivsOut[0], nsd) << std::endl
		      << "set->get derivs difference: " << sse(&checkDerivsDf[0], nsd)
			  << std::endl << std::endl;



	//delete experJan;
	delete experTrain;
	delete experStep;
	delete experSplit1;
	delete experSplit2;
	system ("pause");
}
