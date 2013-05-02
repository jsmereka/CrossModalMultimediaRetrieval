#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mex.h>
#include "topiclib.cpp"

// Syntax
//   [ THETA , MU ] = siBP( WD , phi , N , ALPHA , BETA , SEED , OUTPUT )

// Syntax
//   [ THETA , Mu ] = siBP( WD , phi , N , ALPHA , BETA , SEED , OUTPUT , MUIN )


void siBP(double ALPHA, double BETA, int W, int J, int D, int NN, int OUTPUT, mwIndex *jc, mwIndex *ir, double *sr,
	double *phi, double *theta, double *mu, int startcond) 
{
	int wi, di, i, j, topic, iter;
	double *phitot, *thetad, mutot, perp = 0.0, WBETA = (double) (W*BETA), xi, xitot = 0.0, JALPHA = (double) (J*ALPHA);

	phitot = dvec(J);
	thetad = dvec(D);

	for (wi=0; wi<W; wi++) {
		for (j=0; j<J; j++) {
			phitot[j] += (double) phi[wi*J + j];
		}
	}

	if (startcond == 1) {
		/* start from previously saved state */
		for (di=0; di<D; di++) {
			for (i=jc[di]; i<jc[di + 1]; i++) {
				xi = sr[i];
				xitot += xi;
				thetad[di] += xi;
				for (j=0; j<J; j++) {
					theta[di*J + j] += xi*mu[i*J + j]; // increment theta count matrix
				}
			}
		}
	}

	if (startcond == 0) {
		/* random initialization */
		for (di=0; di<D; di++) {
			for (i=jc[di]; i<jc[di + 1]; i++) {
				xi = sr[i];
				xitot += xi;
				thetad[di] += xi;
				// pick a random topic 0..J-1
				topic = (int) (J*drand());
				mu[i*J + topic] = (double) 1; // assign this word token to this topic
				theta[di*J + topic] += xi; // increment theta count matrix
			}
		}
	}

	for (iter=0; iter<NN; iter++) {

		if (OUTPUT >= 1) {			
			if (((iter % 10)==0) && (iter != 0)) {
				/* calculate perplexity */
				perp = (double) 0;
				for (di=0; di<D; di++) {
					for (i=jc[di]; i<jc[di + 1]; i++) {
						wi = (int) ir[i];
						xi = sr[i];
						mutot = (double) 0;
						for (j=0; j<J; j++) {
							mutot += ((double) phi[wi*J + j] + (double) BETA)/
								((double) phitot[j] + (double) WBETA)*
								((double) theta[di*J + j] + (double) ALPHA)/
								((double) thetad[di] + (double) JALPHA);
						}					
						perp -= (log(mutot) * xi);
					}
				}
				mexPrintf("\tIteration %d of %d:\t%f\n", iter, NN, exp(perp/xitot));
				if ((iter % 10)==0) mexEvalString("drawnow;");
			}
		}

		// iterate all tokens
		for (di=0; di<D; di++) {
			for (i=jc[di]; i<jc[di + 1]; i++) {
				wi = (int) ir[i];
				xi = sr[i];
				mutot = (double) 0;
				for (j=0; j<J; j++) {
					mu[i*J + j] = ((double) phi[wi*J + j] + (double) BETA)/((double) phitot[j] + (double) WBETA)*
						((double) theta[di*J + j] + (double) ALPHA);
					mutot += mu[i*J + j];
				}
				for (j=0; j<J; j++) {
					mu[i*J + j] /= mutot;
				}
			}
		}

		// clear theta and update theta
		for (i=0; i<J*D; i++) theta[i] = (double) 0;
		for (di=0; di<D; di++) {
			for (i=jc[di]; i<jc[di + 1]; i++) {
				xi = sr[i];
				for (j = 0; j < J; j++) { 
					theta[di*J + j] += xi*mu[i*J + j]; // increment theta count matrix
				}
			}
		}
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	double *MUIN, *phi, *theta, *mu, *sr;
	double ALPHA, BETA;
	int W, J, D, NN, SEED, OUTPUT, nzmax, i, j, startcond;
	mwIndex *jc, *ir;

	/* Check for proper number of arguments. */
	if (nrhs < 7) {
		mexErrMsgTxt("At least 7 input arguments required");
	} else if (nlhs < 1) {
		mexErrMsgTxt("At least 1 output arguments required");
	}

	startcond = 0;
	if (nrhs == 8) startcond = 1;

	/* dealing with sparse array WD */
	if (mxIsDouble(prhs[0]) != 1) mexErrMsgTxt("WD must be a double precision matrix");
	sr = mxGetPr(prhs[0]);
	ir = mxGetIr(prhs[0]);
	jc = mxGetJc(prhs[0]);
	nzmax = (int) mxGetNzmax(prhs[0]);
	W = (int) mxGetM(prhs[0]);
	D = (int) mxGetN(prhs[0]);

	phi = mxGetPr(prhs[1]);
	J = (int) mxGetM(prhs[1]);
	if (J<=0) mexErrMsgTxt("Number of topics must be greater than zero");
	if ((int) mxGetN(prhs[1]) != W) mexErrMsgTxt("Vocabulary mismatches");

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

	// seeding
	seedMT(1 + SEED*2); // seeding only works on uneven numbers

	/* allocate memory */
	mu = dvec(J*nzmax);

	if (startcond == 1) {
		for (i=0; i<nzmax; i++) mu[i] = (double) MUIN[i];   
	} 

	theta = dvec(J*D);

	/* run the learning algorithm */
	siBP(ALPHA, BETA, W, J, D, NN, OUTPUT, jc, ir, sr, phi, theta, mu, startcond);

	/* output */
	plhs[0] = mxCreateDoubleMatrix(J, D, mxREAL );
	mxSetPr(plhs[0], theta);

	plhs[1] = mxCreateDoubleMatrix(J, nzmax, mxREAL );
	mxSetPr(plhs[1], mu);
}
