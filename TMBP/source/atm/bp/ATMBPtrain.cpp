#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mex.h>
#include "topiclib.cpp"

#define NAMAX 50 // maximum number of authors on a document  

// Syntax
// [ phi , theta , muz , mux ] = ATMBP( WD , AD , J , N , ALPHA , BETA , SEED , OUTPUT )
// [ phi , theta , muz , mux ] = ATMBP( WD , AD , J , N , ALPHA , BETA , SEED , OUTPUT , MUZIN , MUXIN )

void ATMBP(double ALPHA, double BETA, int W, int J, int D, int A, int MA, int NN, int OUTPUT, 
	mwIndex *irwd, mwIndex *jcwd, double *srwd, mwIndex *irad, mwIndex *jcad, 
	double *muz, double *mux, double *phi, double *theta, int startcond) 
{ 		
	int wi, di, ai, i, j, a, topic, iter;
	double xi, totprob, probs, WBETA = (double) (W*BETA), JALPHA = (double) (J*ALPHA);
	double *thetad, *phitot, *xprob, *zprob;

	phitot = dvec(J);
	thetad = dvec(A);
	xprob = dvec(MA);
	zprob = dvec(J);

	if (startcond==1) {
		/* start from previous state */
		for (di=0; di<D; di++) {
			for (i=jcwd[di]; i<jcwd[di + 1]; i++) {
				wi = (int) irwd[i];
				xi = srwd[i];
				for (j=0; j<J; j++) {
					phi[wi*J + j] += xi*muz[i*J + j]; // increment phi count matrix
					phitot[j] += xi*muz[i*J + j]; // increment phitot matrix
					for (a=0; a<(jcad[di+1] - jcad[di]); a++) {
						ai = (int) irad[jcad[di] + a];
						theta[ai*J + j] += xi*muz[i*J + j]*mux[i*MA + a]; // increment theta count matrix
						thetad[ai] += xi*muz[i*J + j]*mux[i*MA + a];
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
				// pick a random topic 0..J-1
				topic = (int) (J*drand());
				muz[i*J + topic] = (double) 1; // assign this word token to this topic
				phi[wi*J + topic] += xi; // increment phi count matrix
				phitot[topic] += xi; // increment phitot matrix
				/* pick a random number between jcad[di + 1] and jcad[di] */
				a = (int) ((jcad[di + 1] - jcad[di])*drand());
				ai = (int) irad[jcad[di] + a]; // assign this word to this author
				mux[i*MA + a] = (double) 1;
				// update counts for this author
				theta[ai*J + topic] += xi; // increment theta count matrix
				thetad[ai] += xi;
			}
		}
	}

	for (iter=0; iter<NN; iter++) {

		if (OUTPUT >=1) {
			if ((iter % 10)==0) mexPrintf( "\tIteration %d of %d\n" , iter , NN );
			if ((iter % 10)==0) mexEvalString("drawnow;");
		}

		for (di=0; di<D; di++) {
			for (i=jcwd[di]; i<jcwd[di + 1]; i++) {
				wi = (int) irwd[i]; // current word index 
				xi = srwd[i]; // current word counts
				// message
				for (a=0; a<(jcad[di + 1]-jcad[di]); a++) xprob[a] = (double) 0;
				for (j=0; j<J; j++) zprob[j] = (double) 0;
				totprob = (double) 0;
				for (a=0; a<(jcad[di + 1]-jcad[di]); a++) {
					ai = (int) irad[jcad[di] + a]; // current author index under consideration					
					for (j=0; j<J; j++) {	  
						// probs contains the (unnormalized) probability of assigning this word token to topic j and author ai
						probs = ((double) phi[wi*J + j] - (double) xi*muz[i*J + j] + (double) BETA)/ 
							((double) phitot[j] - (double) xi*muz[i*J + j] + (double) WBETA) *	             
							((double) theta[ai*J + j] - (double) xi*muz[i*J + j]*mux[i*MA + a] + (double) ALPHA)/
							((double) thetad[ai] - (double) xi*mux[i*MA + a] + (double) JALPHA);
						xprob[a] += probs;
						zprob[j] += probs;
						totprob += probs;
					}
				}
				for (a=0; a<(jcad[di + 1]-jcad[di]); a++) {
					mux[i*MA + a] = xprob[a]/totprob;
				}
				for (j=0; j<J; j++) {
					muz[i*J + j] = zprob[j]/totprob;
				}
			}
		}

		/* clear phi, theta, thetad and phitot */
		for (i=0; i<J*W; i++) phi[i] = (double) 0;
		for (i=0; i<J*A; i++) theta[i] = (double) 0;
		for (j=0; j<J; j++) phitot[j] = (double) 0;
		for (i=0; i<A; i++) thetad[i] = (double) 0;

		// update parameters
		for (di=0; di<D; di++) {
			for (i=jcwd[di]; i<jcwd[di + 1]; i++) {
				wi = (int) irwd[i];
				xi = srwd[i];
				for (j=0; j<J; j++) {
					phi[wi*J + j] += xi*muz[i*J + j]; // increment phi count matrix
					phitot[j] += xi*muz[i*J + j]; // increment phitot matrix
					for (a=0; a<(jcad[di+1] - jcad[di]); a++) {
						ai = (int) irad[jcad[di] + a];
						theta[ai*J + j] += xi*muz[i*J + j]*mux[i*MA + a]; // increment theta count matrix
						thetad[ai] += xi*muz[i*J + j]*mux[i*MA + a];
					}
				}
			}
		}
	}
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	double *srwd, *srad, *MUZIN, *MUXIN, *theta, *phi, *muz, *mux;
	double ALPHA, BETA;	
	int W, J, D, A, MA = 0, NN, SEED, OUTPUT, nzmaxwd, nzmaxad, i, j, a, startcond;
	mwIndex *irwd, *jcwd, *irad, *jcad;

	/* Check for proper number of arguments. */
	if (nrhs < 8) {
		mexErrMsgTxt("At least 8 input arguments required");
	} else if (nlhs < 2) {
		mexErrMsgTxt("At least 2 output arguments required");
	}

	startcond = 0;
	if (nrhs > 8) startcond = 1;

	/* dealing with sparse array WD */
	if (mxIsDouble(prhs[0]) != 1) mexErrMsgTxt("WD must be a double precision matrix");
	srwd = mxGetPr(prhs[0]);
	irwd = mxGetIr(prhs[0]);
	jcwd = mxGetJc(prhs[0]);
	nzmaxwd = (int) mxGetNzmax(prhs[0]);
	W = (int) mxGetM(prhs[0]);
	D = (int) mxGetN(prhs[0]);

	/* dealing with sparse array AD */
	if (mxIsDouble(prhs[1]) != 1) mexErrMsgTxt("AD must be a double precision matrix");
	srad = mxGetPr(prhs[1]);
	irad = mxGetIr(prhs[1]);
	jcad = mxGetJc(prhs[1]);
	nzmaxad = (int) mxGetNzmax(prhs[1]);
	A = (int) mxGetM(prhs[1]);
	if ((int) mxGetN(prhs[1]) != D) mexErrMsgTxt("WD and AD must have the same number of columns");

	/* check that every document has some authors */
	for (i=0; i<D; i++) {
		if ((jcad[i + 1] - jcad[i]) == 0) mexErrMsgTxt("there are some documents without authors in AD matrix ");
		if ((jcad[i + 1] - jcad[i]) > NAMAX) mexErrMsgTxt("Too many authors in some documents ... reached the NAMAX limit");
		if ((jcad[i + 1] - jcad[i]) > MA) MA = (int) (jcad[i + 1] - jcad[i]);
	}

	J = (int) mxGetScalar(prhs[2]);
	if (J<=0) mexErrMsgTxt("Number of topics must be greater than zero");

	NN = (int) mxGetScalar(prhs[3]);
	if (NN<0) mexErrMsgTxt("Number of iterations must be greater than zero");

	ALPHA = (double) mxGetScalar(prhs[4]);
	if (ALPHA<0) mexErrMsgTxt("ALPHA must be greater than zero");

	BETA = (double) mxGetScalar(prhs[5]);
	if (BETA<0) mexErrMsgTxt("BETA must be greater than zero");

	SEED = (int) mxGetScalar(prhs[6]);
	// set the seed of the random number generator

	OUTPUT = (int) mxGetScalar(prhs[7]);

	if (startcond == 1) {
		MUZIN = mxGetPr(prhs[8]);
		if (nzmaxwd != mxGetN(prhs[8])) mexErrMsgTxt("WD and MUZIN mismatch");
		if (J != mxGetM( prhs[ 8 ])) mexErrMsgTxt("J and MUZIN mismatch");
		MUXIN = mxGetPr(prhs[9]);
		if (nzmaxwd != mxGetN( prhs[9])) mexErrMsgTxt("WD and MUXIN mismatch");
		if (MA != mxGetM(prhs[9])) mexErrMsgTxt("MA and MUXIN mismatch");
	}

	// seeding
	seedMT(1 + SEED * 2); // seeding only works on uneven numbers

	/* allocate memory */
	muz  = dvec(J*nzmaxwd);
	mux  = dvec(MA*nzmaxwd);

	if (startcond == 1) {
		for (i=0; i<J*nzmaxwd; i++) muz[i] = (double) MUZIN[i]; 
		for (a=0; a<MA*nzmaxwd; a++) mux[i] = (double) MUXIN[i];
	}

	phi = dvec(J*W);
	theta = dvec(J*A);

	/* run the model */
	ATMBP(ALPHA, BETA, W, J, D, A, MA, NN, OUTPUT, irwd, jcwd, srwd, irad, jcad, muz, mux, phi, theta, startcond);

	/* output */
	plhs[0] = mxCreateDoubleMatrix(J, W, mxREAL);
	mxSetPr(plhs[0], phi);

	plhs[1] = mxCreateDoubleMatrix(J, A, mxREAL);
	mxSetPr(plhs[1], theta);

	plhs[2] = mxCreateDoubleMatrix(J, nzmaxwd, mxREAL);
	mxSetPr(plhs[2], muz);

	plhs[3] = mxCreateDoubleMatrix(MA, nzmaxwd, mxREAL);
	mxSetPr(plhs[3], mux);
}
