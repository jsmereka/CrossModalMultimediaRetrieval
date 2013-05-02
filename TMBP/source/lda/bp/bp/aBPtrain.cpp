#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mex.h>
#include "topiclib.cpp"

// Syntax
//   [ PHI , THETA , MU ] = aBP( WD , J , N , ALPHA , BETA , SEED , OUTPUT )

// Syntax
//   [ PHI , THETA , MU ] = aBP( WD , J , N , ALPHA , BETA , SEED , OUTPUT , MUIN )


void aBP(double ALPHA, double BETA, int W, int J, int D, int NN, int OUTPUT, double *sr, mwIndex *ir, mwIndex *jc,
	double *phi, double *theta, double *mu, int startcond) 
{
	int wi, di, i, j, topic, iter, *order, rp, temp, ii;
	double *phitot, mutot, *thetad, JALPHA = (double) (J*ALPHA), WBETA = (double) (W*BETA), xi, xitot = 0.0, perp = 0.0;

	phitot = dvec(J);
	thetad = dvec(D);
	order = ivec(D);

	if (startcond == 1) {
		/* start from previously saved state */
		for (di=0; di<D; di++) {
			for (i=jc[di]; i<jc[di + 1]; i++) {
				wi = (int) ir[i];
				xi = sr[i];
				xitot += xi;
				thetad[di] += xi;
				for (j=0; j<J; j++) {
					phi[wi*J + j] += xi*mu[i*J + j]; // increment phi count matrix
					theta[di*J + j] += xi*mu[i*J + j]; // increment theta count matrix	
					phitot[j] += xi*mu[i]; // increment phitot matrix
				}
			}
		}
	}

	if (startcond == 0) {
		/* random initialization */
		for (di=0; di<D; di++) {
			for (i=jc[di]; i<jc[di + 1]; i++) {
				wi = (int) ir[i];
				xi = sr[i];
				thetad[di] += xi;
				xitot += xi;
				// pick a random topic 0..J-1
				topic = (int) (J*drand());
				mu[i*J + topic] = (double) 1; // assign this word token to this topic
				phi[wi*J + topic] += xi; // increment phi count matrix
				theta[di*J + topic] += xi; // increment theta count matrix
				phitot[topic] += xi; // increment phitot matrix
			}
		}
	}

	/* Determine random order */
	for (i=0; i<D; i++) order[i] = i; // fill with increasing series
	for (i=0; i<(D-1); i++) {
		// pick a random integer between i and nw
		rp = i + (int) ((D-i)*drand());
		// switch contents on position i and position rp
		temp = order[rp];
		order[rp] = order[i];
		order[i] = temp;
	}

	for (iter=0; iter<NN; iter++) {

		if (OUTPUT >= 1) {
			if (((iter % 10)==0) && (iter != 0)) {
				/* calculate perplexity */
				perp = (double) 0;
				for (di=0; di<D; di++) {
					for (i=jc[di]; i<jc[di + 1]; i++) {
						wi = (int) ir[i];
						xi = (double) sr[i];
						mutot = (double) 0;
						for (j=0; j<J; j++) {
							mutot += ((double) phi[wi*J + j] + (double) BETA)/
								((double) phitot[j] + (double) WBETA)*
								((double) theta[di*J + j] + (double) ALPHA)/
								((double) thetad[di] + (double) JALPHA);
						}
						perp -= log(mutot)*xi;
					}
				}
				mexPrintf("\tIteration %d of %d:\t%f\n", iter, NN, exp(perp/xitot));
				if ((iter % 10)==0) mexEvalString("drawnow;");
			}
		}

		/* passing message mu */
		for (ii=0; ii<D; ii++) {
			di = (int) order[ii];
			for (i=jc[di]; i<jc[di + 1]; i++) {
				wi = (int) ir[i];
				xi = sr[i];
				mutot = (double) 0;
				for (j=0; j<J; j++) {
					phi[wi*J + j] -= (double) xi*mu[i*J + j];
					phitot[j] -= (double) xi*mu[i*J + j];
					theta[di*J + j] -= (double) xi*mu[i*J + j];
					mu[i*J + j] = ((double) phi[wi*J + j] + (double) BETA)/
						((double) phitot[j] + (double) WBETA)*
						((double) theta[di*J + j] + (double) ALPHA);
					mutot += mu[i*J + j];
				}
				for (j=0; j<J; j++) {
					mu[i*J + j] /= mutot;
					phi[wi*J + j] += (double) xi*mu[i*J + j];
					phitot[j] += (double) xi*mu[i*J + j];
					theta[di*J + j] += (double) xi*mu[i*J + j];
				}
			}
		}
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	double *mu, *MUIN, *phi, *theta, *sr;
	double ALPHA, BETA;
	mwIndex *ir, *jc;
	int W, J, D, NN, SEED, OUTPUT, nzmax, i, j, wi, di, startcond;

	/* Check for proper number of arguments. */
	if (nrhs < 7) {
		mexErrMsgTxt("At least 7 input arguments required");
	} else if (nlhs < 2) {
		mexErrMsgTxt("At least 2 output arguments required");
	}

	startcond = 0;
	if (nrhs == 8) startcond = 1;

	/* read sparse array WD */
	if (mxIsDouble(prhs[0]) != 1) mexErrMsgTxt("WD must be a double precision matrix");
	sr = mxGetPr(prhs[0]);
	ir = mxGetIr(prhs[0]);
	jc = mxGetJc(prhs[0]);
	nzmax = (int) mxGetNzmax(prhs[0]);
	W = (int) mxGetM(prhs[0]);
	D = (int) mxGetN(prhs[0]);

	J = (int) mxGetScalar(prhs[1]);
	if (J<=0) mexErrMsgTxt("Number of topics must be greater than zero");

	NN = (int) mxGetScalar(prhs[2]);
	if (NN<0) mexErrMsgTxt("Number of iterations must be positive");

	ALPHA = (double) mxGetScalar(prhs[3]);
	if (ALPHA<0) mexErrMsgTxt("ALPHA must be greater than zero");

	BETA = (double) mxGetScalar(prhs[4]);
	if (BETA<0) mexErrMsgTxt("BETA must be greater than zero");

	SEED = (int) mxGetScalar(prhs[5]);

	OUTPUT = (int) mxGetScalar(prhs[6]);

	if (startcond == 1) {
		MUIN = mxGetPr(prhs[7]);
		if (nzmax != (mxGetN(prhs[7]))) mexErrMsgTxt("WD and MUIN mismatch");
		if (J != (mxGetM(prhs[7]))) mexErrMsgTxt("J and MUIN mismatch");
	}

	/* seeding */
	seedMT(1 + SEED*2); // seeding only works on uneven numbers

	/* allocate memory */
	mu  = dvec(J*nzmax);
	if (startcond == 1) {
		for (i=0; i<J*nzmax; i++) mu[i] = (double) MUIN[i];   
	}
	phi = dvec(J*W);
	theta = dvec(J*D);

	/* run the learning algorithm */
	aBP(ALPHA, BETA, W, J, D, NN, OUTPUT, sr, ir, jc, phi, theta, mu, startcond);

	/* output */
	plhs[0] = mxCreateDoubleMatrix(J, W, mxREAL);
	mxSetPr(plhs[0], phi);

	plhs[1] = mxCreateDoubleMatrix(J, D, mxREAL );
	mxSetPr(plhs[1], theta);

	plhs[2] = mxCreateDoubleMatrix(J, nzmax, mxREAL );
	mxSetPr(plhs[2], mu);
}
