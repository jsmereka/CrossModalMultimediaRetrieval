#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mex.h>
#include "topiclib.cpp"

// Syntax
//   [ PHI , THETA , MU ] = FBP_doc( WD , J , THESHOLD , N , ALPHA , BETA , SEED , OUTPUT )

// Syntax
//   [ PHI , THETA , MU ] = FBP_doc( WD , J , THESHOLD , N , ALPHA , BETA , SEED , OUTPUT , MUIN )

/* FBP_doc algorithm */
void FBP_doc(double ALPHA, double BETA, int W, int J, int D, int NN, int OUTPUT, int nzmax, 
	double *sr, mwIndex *ir, mwIndex *jc, double *phi, double *theta, double *mu, double threshold, int startcond) 
{
	int wi, di, i, j, k, topic, iter, rp, temp, ii;
	int *order, *indx_r;
	double mutot, totprob, xi, xitot, perp, trap = 1e-6;
	double JALPHA = (double) (J*ALPHA), WBETA = (double) (W*BETA);
	double *phitot, *thetad, *r, *munew;

	phitot = dvec(J);
	thetad = dvec(D);
	order = ivec(D);
	indx_r = ivec(J*D);
	munew = dvec(J);
	r = dvec(J*D);

	/* phitot */
	for (wi=0; wi<W; wi++) {
		for (j=0; j<J; j++) {
			phitot[j] += phi[wi*J + j];
		}
	}

	if (startcond == 1) {
		/* start from previously saved state */
		for (di=0; di<D; di++) {
			for (i=jc[di]; i<jc[di + 1]; i++) {
				wi = (int) ir[i];
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
				wi = (int) ir[i];
				xi = sr[i];
				thetad[di] += xi;
				xitot += xi;
				// pick a random topic 0..J-1
				topic = (int) (J*drand());
				mu[i*J + topic] = 1.0; // assign this word token to this topic
				theta[di*J + topic] += xi; // increment theta count matrix
			}
		}
	}

	/* Determine random order */
	for (i=0; i<D; i++) order[i] = i; // fill with increasing series
	for (i=0; i<(D-1); i++) {
		// pick a random integer between i and D
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
				perp = 0.0;
				for (di=0; di<D; di++) {
					for (i=jc[di]; i<jc[di + 1]; i++) {
						wi = (int) ir[i];
						xi = sr[i];
						mutot = 0.0;
						for (j=0; j<J; j++) {
							mutot += (phi[wi*J + j] + BETA)/(phitot[j] + WBETA)*
								(theta[di*J + j] + ALPHA)/(thetad[di] + JALPHA);
						}
						perp -= (log(mutot)*xi);
					}
				}
				mexPrintf("\tIteration %d of %d:\t%f\n", iter, NN, exp(perp/xitot));
				if ((iter % 10)==0) mexEvalString("drawnow;");
			}
		}

		/* passing message mu */
		
		/* iteration 0 */
		if (iter == 0) {
			for (ii=0; ii<D; ii++) {
				di = (int) order[ii];
				for (i=jc[di]; i<jc[di + 1]; i++) {
					wi = (int) ir[i];
					xi = sr[i];
					mutot = 0;
					for (j=0; j<J; j++) {
						theta[di*J + j] -= xi*mu[i*J + j];	
						munew[j] = (phi[wi*J + j] + BETA)/(phitot[j] + WBETA)*(theta[di*J + j] + ALPHA);
						mutot += munew[j];
					}
					for (j=0; j<J; j++) {
						munew[j] /= mutot;
						r[di*J + j] += xi*fabs(munew[j] - mu[i*J + j]);
						mu[i*J + j] = munew[j];
						theta[di*J + j] += xi*mu[i*J + j];
					}
				}
				dsort(J, r + di*J, -1, indx_r + di*J);
			}
		} else { /* iteration > 0 */
			for (ii=0; ii<D; ii++) {
				di = (int) order[ii];
				for (j=0; j < (int) (J*threshold); j++) {
					k = (int) indx_r[di*J + j];
					//if (r[di*J + k]*J < trap) break;
					r[di*J + k] = 0.0;
				}
				for (i=jc[di]; i<jc[di + 1]; i++) {
					wi = (int) ir[i];
					xi = sr[i];
					mutot = 0.0;
					totprob = 0.0;
					for (j=0; j < (int) (J*threshold); j++) {
						k = (int) indx_r[di*J + j];
						//if (r[di*J + k]*J < trap) break;
						theta[di*J + k] -= xi*mu[i*J + k];	
						totprob += mu[i*J + k];
						munew[k] = (phi[wi*J + k] + BETA)/(phitot[k] + WBETA)*(theta[di*J + k] + ALPHA);
						mutot += munew[k];
					}
					for (j=0; j < (int) (J*threshold); j++) {
						k = (int) indx_r[di*J + j];
						//if (r[di*J + k]*J < trap) break;
						munew[k] /= mutot;
						munew[k] *= totprob;
						r[di*J + k] += xi*fabs(munew[k] - mu[i*J + k]);
						mu[i*J + k] = munew[k];
						theta[di*J + k] += xi*mu[i*J + k];
					}
				}
				insertionsort(J, r + di*J, indx_r + di*J);
			}
		}
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	double *mu, *MUIN, *phi, *theta, *sr;
	double ALPHA, BETA, threshold;
	mwIndex *ir, *jc;
	int W, J, D, NN, SEED, OUTPUT, nzmax, i, startcond;

	/* Check for proper number of arguments. */
	if (nrhs < 8) {
		mexErrMsgTxt("At least 8 input arguments required");
	} else if (nlhs < 1) {
		mexErrMsgTxt("At least 1 output arguments required");
	}

	startcond = 0;
	if (nrhs == 9) startcond = 1;

	/* read sparse array WD */
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

	threshold = (double) mxGetScalar(prhs[2]);
	if (threshold<0 || threshold>1) mexErrMsgTxt("Threshold should be between 0 and 1.");

	NN = (int) mxGetScalar(prhs[3]);
	if (NN<0) mexErrMsgTxt("Number of iterations must be positive");

	ALPHA = (double) mxGetScalar(prhs[4]);
	if (ALPHA<0) mexErrMsgTxt("ALPHA must be greater than zero");

	BETA = (double) mxGetScalar(prhs[5]);
	if (BETA<0) mexErrMsgTxt("BETA must be greater than zero");

	SEED = (int) mxGetScalar(prhs[6]);

	OUTPUT = (int) mxGetScalar(prhs[7]);

	if (startcond == 1) {
		MUIN = mxGetPr(prhs[8]);
		if (nzmax != (mxGetN(prhs[8]))) mexErrMsgTxt("WD and MUIN mismatch");
		if (J != (mxGetM(prhs[8]))) mexErrMsgTxt("J and MUIN mismatch");
	}

	/* seeding */
	seedMT(1 + SEED*2); // seeding only works on uneven numbers

	/* allocate memory */
	mu  = dvec(J*nzmax);
	if (startcond == 1) {
		for (i=0; i<J*nzmax; i++) mu[i] = (double) MUIN[i];   
	}
	
	theta = dvec(J*D);

	/* run the learning algorithm */
	FBP_doc(ALPHA, BETA, W, J, D, NN, OUTPUT, nzmax, sr, ir, jc, phi, theta, mu, threshold, startcond);

	/* output */
	plhs[0] = mxCreateDoubleMatrix(J, D, mxREAL);
	mxSetPr(plhs[0], theta);

	plhs[1] = mxCreateDoubleMatrix(J, nzmax, mxREAL);
	mxSetPr(plhs[1], mu);
}

