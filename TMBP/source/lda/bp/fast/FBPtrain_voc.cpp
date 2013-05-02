#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mex.h>
#include "topiclib.cpp"

// Syntax
//   [ PHI , THETA , MU ] = FBP_voc( DW , J , THESHOLD , N , ALPHA , BETA , SEED , OUTPUT )

// Syntax
//   [ PHI , THETA , MU ] = FBP_voc( DW , J , THESHOLD , N , ALPHA , BETA , SEED , OUTPUT , MUIN )

/* FBP_voc algorithm */
void FBP_voc(double ALPHA, double BETA, int W, int J, int D, int NN, int OUTPUT, 
	double *sr, mwIndex *ir, mwIndex *jc, double *phi, double *theta, double *mu, double threshold, int startcond) 
{
	int wi, di, i, j, k, topic, iter, rp, temp, ii;
	int *order, *indx_r;
	double mutot, totprob, xi, xitot, perp, trap = 1e-6;
	double JALPHA = (double) (J*ALPHA), WBETA = (double) (W*BETA);
	double *phitot, *thetad, *r, *munew;

	phitot = dvec(J);
	thetad = dvec(D);
	order = ivec(W);
	indx_r = ivec(J*W);
	munew = dvec(J);
	r = dvec(J*W);

	if (startcond == 1) {
		/* start from previously saved state */
		for (wi=0; wi<W; wi++) {
			for (i=jc[wi]; i<jc[wi + 1]; i++) {
				di = (int) ir[i];
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
		for (wi=0; wi<W; wi++) {
			for (i=jc[wi]; i<jc[wi + 1]; i++) {
				di = (int) ir[i];
				xi = sr[i];
				thetad[di] += xi;
				xitot += xi;
				// pick a random topic 0..J-1
				topic = (int) (J*drand());
				mu[i*J + topic] = 1.0; // assign this word token to this topic
				phi[wi*J + topic] += xi; // increment phi count matrix
				theta[di*J + topic] += xi; // increment theta count matrix
				phitot[topic] += xi; // increment phitot matrix
			}
		}
	}

	/* Determine random order */
	for (i=0; i<W; i++) order[i] = i; // fill with increasing series
	for (i=0; i<(W-1); i++) {
		// pick a random integer between i and D
		rp = i + (int) ((W-i)*drand());
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
				for (wi=0; wi<W; wi++) {
					for (i=jc[wi]; i<jc[wi + 1]; i++) {
						di = (int) ir[i];
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
			for (ii=0; ii<W; ii++) {
				wi = (int) order[ii];
				for (i=jc[wi]; i<jc[wi + 1]; i++) {
					di = (int) ir[i];
					xi = sr[i];
					mutot = 0;
					for (j=0; j<J; j++) {
						phi[wi*J + j] -= xi*mu[i*J + j];
						phitot[j] -= xi*mu[i*J + j];
						theta[di*J + j] -= xi*mu[i*J + j];	
						munew[j] = (phi[wi*J + j] + BETA)/(phitot[j] + WBETA)*(theta[di*J + j] + ALPHA);
						mutot += munew[j];
					}
					for (j=0; j<J; j++) {
						munew[j] /= mutot;
						r[wi*J + j] += xi*fabs(munew[j] - mu[i*J + j]);
						mu[i*J + j] = munew[j];
						phi[wi*J + j] += xi*mu[i*J + j];
						phitot[j] += xi*mu[i*J + j];
						theta[di*J + j] += xi*mu[i*J + j];
					}
				}
				dsort(J, r + wi*J, -1, indx_r + wi*J);
			}
		} else { /* iteration > 0 */
			for (ii=0; ii<W; ii++) {
				wi = (int) order[ii];
				for (j=0; j < (int) (J*threshold); j++) {				
					k = (int) indx_r[wi*J + j];
					r[wi*J + k] = 0.0;
				}
				for (i=jc[wi]; i<jc[wi + 1]; i++) {
					di = (int) ir[i];
					xi = sr[i];
					mutot = 0.0;
					totprob = 0.0;
					for (j=0; j < (int) (J*threshold); j++) {
						k = (int) indx_r[wi*J + j];
						phi[wi*J + k] -= xi*mu[i*J + k];
						phitot[k] -= xi*mu[i*J + k];
						theta[di*J + k] -= xi*mu[i*J + k];	
						totprob += mu[i*J + k];
						munew[k] = (phi[wi*J + k] + BETA)/(phitot[k] + WBETA)*(theta[di*J + k] + ALPHA);
						mutot += munew[k];
					}
					for (j=0; j < (int) (J*threshold); j++) {
						k = (int) indx_r[wi*J + j];
						munew[k] /= mutot;
						munew[k] *= totprob;
						r[wi*J + k] += xi*fabs(munew[k] - mu[i*J + k]);
						mu[i*J + k] = munew[k];
						phi[wi*J + k] += xi*mu[i*J + k];
						phitot[k] += xi*mu[i*J + k];
						theta[di*J + k] += xi*mu[i*J + k];
					}
				}
				insertionsort(J, r + wi*J, indx_r + wi*J);
			}
			if (iter > NN - 10) {
				mexPrintf("\tIter %d  Sec %.2f\n", iter, etime());
				mexEvalString("drawnow;");
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
	} else if (nlhs < 2) {
		mexErrMsgTxt("At least 2 output arguments required");
	}

	startcond = 0;
	if (nrhs == 9) startcond = 1;

	/* read sparse array DW */
	if (mxIsDouble(prhs[0]) != 1) mexErrMsgTxt("DW must be a double precision matrix");
	sr = mxGetPr(prhs[0]);
	ir = mxGetIr(prhs[0]);
	jc = mxGetJc(prhs[0]);
	nzmax = (int) mxGetNzmax(prhs[0]);
	D = (int) mxGetM(prhs[0]);
	W = (int) mxGetN(prhs[0]);

	J = (int) mxGetScalar(prhs[1]);
	if (J<=0) mexErrMsgTxt("Number of topics must be greater than zero");

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
		if (nzmax != (mxGetN(prhs[8]))) mexErrMsgTxt("DW and MUIN mismatch");
		if (J != (mxGetM(prhs[8]))) mexErrMsgTxt("J and MUIN mismatch");
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
	FBP_voc(ALPHA, BETA, W, J, D, NN, OUTPUT, sr, ir, jc, phi, theta, mu, threshold, startcond);

	/* output */
	plhs[0] = mxCreateDoubleMatrix(J, W, mxREAL);
	mxSetPr(plhs[0], phi);

	plhs[1] = mxCreateDoubleMatrix(J, D, mxREAL);
	mxSetPr(plhs[1], theta);

	plhs[2] = mxCreateDoubleMatrix(J, nzmax, mxREAL);
	mxSetPr(plhs[2], mu);
}
