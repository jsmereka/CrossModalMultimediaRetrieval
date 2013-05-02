#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mex.h>
#include "topiclib.cpp"

// Syntax
//   [ theta , mu ] = aBP( WD , phi , N , ALPHA , BETA , SEED , OUTPUT )

// Syntax
//   [ theta , mu ] = aBP( WD , phi , N , ALPHA , BETA , SEED , OUTPUT , MUIN )


void aBP( double ALPHA, double BETA, int W, int J, int D, int NN, int OUTPUT, mwIndex *jc, mwIndex *ir, double *sr,
	double *phi, double *theta, double *mu, int startcond ) {

		int wi, di, i, j, topic, iter, *order, rp, temp, ii;
		double *phitot, mutot, *thetad, JALPHA = (double) (J*ALPHA), WBETA = (double) (W*BETA), xi, xitot = 0.0, perp = 0.0;

		phitot = dvec(J);
		thetad = dvec(D);
		order = ivec(D);

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

		/* Determine random order */
		for (i=0; i<D; i++) order[i] = i; // fill with increasing series
		for (i=0; i<(D-1); i++) {
			// pick a random integer between i and nw
			rp = i + (int) ((D-i)*drand());
			// switch contents on position i and position rp
			temp = order[rp];
			order[rp]=order[i];
			order[i]=temp;
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
									((double) thetad[di] + (double) JALPHA);;
							}
							perp -= log(mutot)*xi;
						}
					}
					mexPrintf("\tIteration %d of %d:\t%f\n", iter, NN, exp(perp/xitot));
					if ((iter % 10)==0) mexEvalString("drawnow;");
				}
			}

			// iterate all tokens
			for (ii=0; ii<D; ii++) {
				di = (int) order[ii];
				for (i=jc[di]; i<jc[di + 1]; i++) {
					wi = (int) ir[i];
					xi = sr[i];
					mutot = (double) 0;
					for (j=0; j<J; j++) {
						theta[di*J + j] -= (double) xi*mu[i*J + j];
						mu[i*J + j] = ((double) phi[wi*J + j] + (double) BETA)/
							((double) phitot[j] + (double) WBETA)*
							((double) theta[di*J + j] + (double) ALPHA);
						mutot += mu[i*J + j];
					}
					for (j=0; j<J; j++) {
						mu[i*J + j] /= mutot;
						theta[di*J + j] += (double) xi*mu[i*J + j];
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
	mu  = dvec(J*nzmax);
	if (startcond == 1) {
		for (i=0; i<nzmax; i++) mu[i] = (double) MUIN[i];   
	} 

	theta = dvec(J*D);

	/* run the learning algorithm */
	aBP(ALPHA, BETA, W, J, D, NN, OUTPUT, jc, ir, sr, phi, theta, mu, startcond);

	/* output */
	plhs[0] = mxCreateDoubleMatrix(J, D, mxREAL );
	mxSetPr(plhs[0], theta);

	plhs[1] = mxCreateDoubleMatrix(J, nzmax, mxREAL );
	mxSetPr(plhs[1], mu);
}
