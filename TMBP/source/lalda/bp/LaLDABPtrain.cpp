#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mex.h>
#include "topiclib.cpp"

#define NAMAX 50 // maximum number of labels on a document  

// Syntax
// [ PHI , THETA , Z ] = LaLDAGS( WD , AD , N , ALPHA , BETA , SEED , OUTPUT )
// [ PHI , THETA , Z ] = LaLDAGS( WD , AD , N , ALPHA , BETA , SEED , OUTPUT , ZIN )

void LaLDAGS(double ALPHA, double BETA, int W, int D, int A, int MA, int NN, int OUTPUT, 
	mwIndex *irwd, mwIndex *jcwd, double *srwd, mwIndex *irad, mwIndex *jcad, 
	int *z, double *phi, double *theta, int nzmaxad, int startcond) 
{ 
	int wi, di, ai, i, a, topic, iter;
	double xi, totprob, WBETA = (double) (W*BETA), max;
	double *phitot, *theta2, *phi2, *phitot2, *zprobs;

	phitot = dvec(A);
	phitot2 = dvec(A);
	theta2 = dvec(nzmaxad);
	phi2 = dvec(A*W);
	zprobs = dvec(MA);

	if (startcond==1) {
		/* start from previous state */
		for (di=0; di<D; di++) {
			for (i=jcwd[di]; i<jcwd[di + 1]; i++) {
				wi = (int) irwd[i];
				xi = srwd[i];
				phi[wi*A + z[i]] += xi; // increment phi count matrix
				phitot[z[i]] += xi; // increment phitot matrix
				for (a=jcad[di]; a<jcad[di + 1]; a++) {
					ai = (int) irad[a];
					if (ai == z[i]) {
						theta[a] += xi; // increment theta count matrix
					}
				}				
			}
		}
	}

	if (startcond==0) {
		/* random initialization */
		if (OUTPUT==2) mexPrintf( "Starting Random initialization\n" );
		for (di=0; di<D; di++) {
			for (i=jcwd[di]; i<jcwd[di + 1]; i++) {
				wi = (int) irwd[i];
				xi = srwd[i];
				// pick a random topic or tag 0..XA
				topic = (int) ((jcad[di + 1] - jcad[di])*drand());
				ai = irad[jcad[di] + topic];
				z[i] = (int) ai; // assign this word token to this topic
				phi[wi*A + ai] += xi; // increment phi count matrix
				phitot[ai] += xi; // increment phitot matrix	
				theta[jcad[di] + topic] += xi; // increment theta count matrix
			}
		}
	}

	for (iter=0; iter<NN; iter++) {

		if (OUTPUT >=1) {
			if ((iter % 10)==0) mexPrintf("\tIteration %d of %d\n", iter, NN);
			if ((iter % 10)==0) mexEvalString("drawnow;");
		}

		for (i=0; i<A*W; i++) phi2[i] = (double) 0;
		for (i=0; i<nzmaxad; i++) theta2[i] = (double) 0;
		for (a=0; a<A; a++) phitot2[a] = (double) 0;

		for (di=0; di<D; di++) {
			for (i=jcwd[di]; i<jcwd[di + 1]; i++) {
				wi = (int) irwd[i]; // current word index 
				xi = srwd[i]; // current word counts
				// message
				totprob = (double) 0;
				for (a=jcad[di]; a<jcad[di + 1]; a++) {
					ai = (int) irad[a]; // current label index under consideration					  
					// probs contains the (unnormalized) probability of assigning this word token to topic j and label ai
					zprobs[a-jcad[di]] = ((double) phi[wi*A + ai] + (double) BETA)/((double) phitot[ai] + (double) WBETA) *	             
						((double) theta[a] + (double) ALPHA);
					totprob += zprobs[a-jcad[di]];
				}
				max = zprobs[0];
				for (a=0; a<(jcad[di + 1]-jcad[di]); a++) {
					zprobs[a] /= totprob;
					if (zprobs[a] > max) {
						max = zprobs[a];
						z[i] = (int) irad[jcad[di] + a];
					}
				}
				for (a=jcad[di]; a<jcad[di + 1]; a++) {
					ai = (int) irad[a];			  
					phi2[wi*A + ai] += zprobs[a-jcad[di]]*xi;
					phitot2[ai] += zprobs[a-jcad[di]]*xi;
					theta2[a] += zprobs[a-jcad[di]]*xi;
				}				
			}
		}

		/* copy phi, theta, thetad and phitot */
		for (i=0; i<A*W; i++) phi[i] = (double) phi2[i];
		for (i=0; i<nzmaxad; i++) theta[i] = (double) theta2[i];
		for (a=0; a<A; a++) phitot[a] = (double) phitot2[a];
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	double *srwd, *srad, *theta, *srtheta, *phi, *Z, *ZIN;
	double ALPHA, BETA;	
	int *z;
	int W, J, D, A, MA = 0, NN, SEED, OUTPUT, nzmaxwd, nzmaxad, i, a, di, startcond;
	mwIndex *irwd, *jcwd, *irad, *jcad;
	mwIndex *irtheta, *jctheta;

	/* Check for proper number of arguments. */
	if (nrhs < 7) {
		mexErrMsgTxt("At least 7 input arguments required");
	} else if (nlhs < 2) {
		mexErrMsgTxt("At least 2 output arguments required");
	}

	startcond = 0;
	if (nrhs > 7) startcond = 1;

	/* dealing with sparse array WD */
	if (mxIsDouble(prhs[0]) != 1) mexErrMsgTxt("WD must be a double prexision matrix");
	srwd = mxGetPr(prhs[0]);
	irwd = mxGetIr(prhs[0]);
	jcwd = mxGetJc(prhs[0]);
	nzmaxwd = (int) mxGetNzmax(prhs[0]);
	W = (int) mxGetM(prhs[0]);
	D = (int) mxGetN(prhs[0]);

	/* dealing with sparse array AD */
	if (mxIsDouble(prhs[1]) != 1) mexErrMsgTxt("AD must be a double prexision matrix");
	srad = mxGetPr(prhs[1]);
	irad = mxGetIr(prhs[1]);
	jcad = mxGetJc(prhs[1]);
	nzmaxad = (int) mxGetNzmax(prhs[1]);
	A = (int) mxGetM(prhs[1]);
	if ((int) mxGetN(prhs[1]) != D) mexErrMsgTxt("WD and AD must have the same number of columns");

	/* check that every document has some authors */
	for (i=0; i<D; i++) {
		if ((jcad[i + 1] - jcad[i]) == 0) mexErrMsgTxt("there are some documents without authors in AD matrix ");
		if ((jcad[i + 1] - jcad[i]) > NAMAX) mexErrMsgTxt("Too many labels in some documents ... reached the NAMAX limit");
		if ((jcad[i + 1] - jcad[i]) > MA) MA = (int) (jcad[i + 1] - jcad[i]);
	}

	NN = (int) mxGetScalar(prhs[2]);
	if (NN<0) mexErrMsgTxt("Number of iterations must be greater than zero");

	ALPHA = (double) mxGetScalar(prhs[3]);
	if (ALPHA<0) mexErrMsgTxt("ALPHA must be greater than zero");

	BETA = (double) mxGetScalar(prhs[4]);
	if (BETA<0) mexErrMsgTxt("BETA must be greater than zero");

	SEED = (int) mxGetScalar(prhs[5]);
	// set the seed of the random number generator

	OUTPUT = (int) mxGetScalar(prhs[6]);

	if (startcond == 1) {
		ZIN = mxGetPr(prhs[7]);
		if (nzmaxwd != mxGetN(prhs[7])) mexErrMsgTxt("WD and ZIN mismatch");
	}

	// seeding
	seedMT(1 + SEED * 2); // seeding only works on uneven numbers

	/* allocate memory */
	z = ivec(nzmaxwd);

	if (startcond == 1) {
		for (i=0; i<J*nzmaxwd; i++) z[i] = (int) ZIN[i] - 1; 
	}

	phi = dvec(A*W);
	theta = dvec(nzmaxad);

	/* run the model */
	LaLDAGS(ALPHA, BETA, W, D, A, MA, NN, OUTPUT, irwd, jcwd, srwd, irad, jcad, z, phi, theta, nzmaxad, startcond);

	/* output */
	plhs[0] = mxCreateDoubleMatrix(A, W, mxREAL);
	mxSetPr(plhs[0], phi);

	/* MAKE theta SPARSE MATRIX */
	plhs[1] = mxCreateSparse(A, D, nzmaxad, mxREAL);
	srtheta  = mxGetPr(plhs[1]);
	irtheta = mxGetIr(plhs[1]);
	jctheta = mxGetJc(plhs[1]);
	for (di=0; di<D; di++) {
		jctheta[di] = jcad[di];
		for (a=jcad[di]; a<jcad[di + 1]; a++) {
			irtheta[a] = irad[a];
		}
	}
	jctheta[D] = jcad[D];
	for (i=0; i<nzmaxad; i++) srtheta[i] = theta[i]; 

	plhs[2] = mxCreateDoubleMatrix(1, nzmaxwd, mxREAL);
	Z = (double *) mxCalloc(nzmaxwd, sizeof(double));
	for (i=0; i<nzmaxwd; i++) Z[i] = (double) z[i] + 1;
	mxSetPr(plhs[2], Z);
}
