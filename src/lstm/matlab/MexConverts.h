#include "matrix.h"
#include <vector>
#include <typeinfo>

// Author: Leo Pape 20101223

#ifndef MEXCONVERTS_H_
#define MEXCONVERTS_H_

// fast mxArray initialization (without zero-filling)
inline mxArray* fastMxArrayInit(int nrow, int ncol) {
    mxArray *result;
    result = mxCreateDoubleMatrix(0, 0, mxREAL);
    mxSetM(result, nrow);
    mxSetN(result, ncol);
    mxSetData(result, mxMalloc(sizeof(double)*nrow*ncol));
	//result = mxCreateDoubleMatrix(ncol, nrow, mxREAL); // slow way of creating array
    return result;
}

// convert std::vector<T> to 1xNCOL mxArray
template <class T>
        static mxArray* v2mxArray(const std::vector<T>& vect) {
    if (vect.empty()) return mxCreateDoubleMatrix(0, 0, mxREAL);
    
    size_t nrow = 1;
    size_t ncol = vect.size();
    
    mxArray *result = fastMxArrayInit(nrow, ncol);
    
    double* tgt_ptr = (double*)mxGetPr(result);
    for (size_t i = 0; i < ncol; i++) {
        *tgt_ptr++ = (double)vect[i];
	}
    
    return result;
}

// convert std::vector<std::vector <T> > to NROW * NCOL mxArray
template <class T>
        static mxArray* vv2mxArray(const std::vector<std::vector<T> >& vect) {
    if (vect.empty()) return mxCreateDoubleMatrix(0, 0, mxREAL);
    
    size_t nrow = vect.size();
    size_t ncol = vect[0].size();
    
    mxArray *result = fastMxArrayInit(nrow, ncol);
    
    double* tgt_ptr = (double*)mxGetPr(result);
    for (size_t c = 0; c < ncol; c++) {
        for (size_t r = 0; r < nrow; r++) {
            *tgt_ptr++ = (double)vect[r][c];
		}
	}
    
    return result;
}



template <class T>
        static std::vector<T> mxArray2v(const mxArray* mxPtr) {
    if (!mxIsDouble(mxPtr)) { mexErrMsgTxt("Conversion only possible from <double> array. Please convert your inputs."); }

    int nelm = mxGetNumberOfElements(mxPtr);
    
    double* src_ptr = (double*)mxGetPr(mxPtr);
    std::vector<T> result(nelm);

	// we cannot overload by return type alone, so need explicit cast for <bool> type
	if ( typeid(T) == typeid(true) ) {
		for (int i = 0; i < nelm; i++) {
	        result[i] = (*src_ptr++) != 0.0;
		}
	} else {
		for (int i = 0; i < nelm; i++) {
		    result[i] = (T) (*src_ptr++);
		}
	}

    return result;
}


// convert NROW * NCOL mxArray to std::vector<std::vector <T> >
template <class T>
        static std::vector< std::vector<T> > mxArray2vv(const mxArray* mxPtr) {
    if (!mxIsDouble(mxPtr)) { mexErrMsgTxt("Conversion only possible for <double> type. Please convert your inputs."); }

    int nrow = mxGetM(mxPtr);
    int ncol = mxGetN(mxPtr);
    
    double* src_ptr = (double*)mxGetPr(mxPtr);
    std::vector< std::vector<T> > result(nrow, std::vector<T>(ncol));

	// we cannot overload by return type alone, so need explicit cast for <bool> type
	if ( typeid(T) == typeid(true) ) {
		for (int c = 0; c < ncol; c++) {
		    for (int r = 0; r < nrow; r++) {
			    result[r][c] = (*src_ptr++) != 0.0;
			}
		}
	} else {
		for (int c = 0; c < ncol; c++) {
			for (int r = 0; r < nrow; r++) {
				result[r][c] = (T)(*src_ptr++);
			}
		}
	}

	return result;
}

template <class T>
        static void printV(const std::vector<T>& vect) {
    size_t ncol = vect.size();
    
    if (ncol<1)
        mexPrintf("Empty vector");

    for (size_t c = 0; c < ncol; c++)
        mexPrintf("%f ", (double) vect[c]);

    mexPrintf("\n");
}

template <class T>
        static void printVV(const std::vector< std::vector<T> >& vect) {
    size_t nrow = vect.size();
    size_t ncol = vect[0].size();
    
    if (nrow < 1)
        mexPrintf("Empty vector");

    for (size_t r = 0; r < nrow; r++) {
        for (size_t c = 0; c < ncol; c++) {
            mexPrintf("%f ", (double) vect[r][c]);
        }
        mexPrintf("\n");
    }

    mexPrintf("\n");
}

#endif /* MEXCONVERTS_H_ */