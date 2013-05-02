#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mex.h>
#include "topiclib.cpp"

// Syntax
//   [ THETA , MU ] = ABP_voc( DW , PHI , TW , TK , N , ALPHA , BETA , SEED , OUTPUT )

// Syntax
//   [ THETA , MU ] = ABP_voc( DW , PHI , TW , TK , N , ALPHA , BETA , SEED , OUTPUT , MUIN )

/* ABP_voc algorithm */
void ABP_voc(double ALPHA, double BETA, int W, int J, int D, int NN, int OUTPUT, 
	double *sr, mwIndex *ir, mwIndex *jc, double *phi, double *theta, double *mu, 
	double tw, double tk, int startcond) 
{
	int wi, di, i, j, k, topic, iter, ii;
	int *ind_rk, *ind_rw;
	double mutot, totprob, xi, xitot, perp;
	double JALPHA = (double) (J*ALPHA), WBETA = (double) (W*BETA);
	double *phitot, *thetad, *rk, *rw, *munew;

	phitot = dvec(J);
	thetad = dvec(D);
	ind_rk = ivec(J*W);
	ind_rw = ivec(W);
	munew = dvec(J);
	rk = dvec(J*W);
	rw = dvec(W);

	/* phitot */
	for (wi=0; wi<W; wi++) {
		for (j=0; j<J; j++) {
			phitot[j] += phi[wi*J + j];
		}
	}

	if (startcond == 1) {
		/* start from previously saved state */
		for (wi=0; wi<W; wi++) {
			for (i=jc[wi]; i<jc[wi + 1]; i++) {
				di = (int) ir[i];
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
		for (wi=0; wi<W; wi++) {
			for (i=jc[wi]; i<jc[wi + 1]; i++) {
				di = (int) ir[i];
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
			for (wi=0; wi<W; wi++) {
				for (i=jc[wi]; i<jc[wi + 1]; i++) {
					di = (int) ir[i];
					xi = sr[i];
					mutot = 0;
					for (j=0; j<J; j++) {
						theta[di*J + j] -= xi*mu[i*J + j];	
						munew[j] = (phi[wi*J + j] + BETA)/(phitot[j] + WBETA)*(theta[di*J + j] + ALPHA);
						mutot += munew[j];
					}
					for (j=0; j<J; j++) {
						munew[j] /= mutot;
						rk[wi*J + j] += xi*fabs(munew[j] - mu[i*J + j]);
						rw[wi] += xi*fabs(munew[j] - mu[i*J + j]);
						mu[i*J + j] = munew[j];
						theta[di*J + j] += xi*mu[i*J + j];
					}
				}
				dsort(J, rk + wi*J, -1, ind_rk + wi*J);
			}
			dsort(W, rw, -1, ind_rw);

		} else { /* iteration > 0 */
			for (ii=0; ii < (int) (tw*W); ii++) {
				wi = (int) ind_rw[ii];
				for (j=0; j < (int) (J*tk); j++) {				
					k = (int) ind_rk[wi*J + j];
					rw[wi] -= rk[wi*J + k];
					rk[wi*J + k] = 0.0;
				}
				for (i=jc[wi]; i<jc[wi + 1]; i++) {
					di = (int) ir[i];
					xi = sr[i];
					mutot = 0.0;
					totprob = 0.0;
					for (j=0; j < (int) (J*tk); j++) {
						k = (int) ind_rk[wi*J + j];
						theta[di*J + k] -= xi*mu[i*J + k];	
						totprob += mu[i*J + k];
						munew[k] = (phi[wi*J + k] + BETA)/(phitot[k] + WBETA)*(theta[di*J + k] + ALPHA);
						mutot += munew[k];
					}
					for (j=0; j < (int) (J*tk); j++) {
						k = (int) ind_rk[wi*J + j];
						munew[k] /= mutot;
						munew[k] *= totprob;
						rk[wi*J + k] += xi*fabs(munew[k] - mu[i*J + k]);
						mu[i*J + k] = munew[k];
						theta[di*J + k] += xi*mu[i*J + k];
					}
				}
				for (j=0; j < (int) (J*tk); j++) {
					k = (int) ind_rk[wi*J + j];
					rw[wi] += rk[wi*J + k];
				}
				insertionsort(J, rk + wi*J, ind_rk + wi*J);
			}

			insertionsort(W, rw, ind_rw);
		}
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	double *mu, *MUIN, *phi, *theta, *sr;
	double ALPHA, BETA, tw, tk;
	mwIndex *ir, *jc;
	int W, J, D, NN, SEED, OUTPUT, nzmax, i, startcond;

	/* Check for proper number of arguments. */
	if (nrhs < 9) {
		mexErrMsgTxt("At least 9 input arguments required");
	} else if (nlhs < 1) {
		mexErrMsgTxt("At least 1 output arguments required");
	}

	startcond = 0;
	if (nrhs == 10) startcond = 1;

	/* read sparse array DW */
	if (mxIsDouble(prhs[0]) != 1) mexErrMsgTxt("DW must be a double precision matrix");
	sr = mxGetPr(prhs[0]);
	ir = mxGetIr(prhs[0]);
	jc = mxGetJc(prhs[0]);
	nzmax = (int) mxGetNzmax(prhs[0]);
	D = (int) mxGetM(prhs[0]);
	W = (int) mxGetN(prhs[0]);

	phi = mxGetPr(prhs[1]);
	J = (int) mxGetM(prhs[1]);
	if (J<=0) mexErrMsgTxt("Number of topics must be greater than zero");
	if ((int) mxGetN(prhs[1]) != W) mexErrMsgTxt("Vocabulary mismatches");

	tw = (double) mxGetScalar(prhs[2]);
	if (tw<0 || tw>1) mexErrMsgTxt("Threshold should be between 0 and 1.");

	tk = (double) mxGetScalar(prhs[3]);
	if (tk<0 || tk>1) mexErrMsgTxt("Threshold should be between 0 and 1.");

	NN = (int) mxGetScalar(prhs[4]);
	if (NN<0) mexErrMsgTxt("Number of iterations must be positive");

	ALPHA = (double) mxGetScalar(prhs[5]);
	if (ALPHA<0) mexErrMsgTxt("ALPHA must be greater than zero");

	BETA = (double) mxGetScalar(prhs[6]);
	if (BETA<0) mexErrMsgTxt("BETA must be greater than zero");

	SEED = (int) mxGetScalar(prhs[7]);

	OUTPUT = (int) mxGetScalar(prhs[8]);

	if (startcond == 1) {
		MUIN = mxGetPr(prhs[9]);
		if (nzmax != (mxGetN(prhs[9]))) mexErrMsgTxt("DW and MUIN mismatch");
		if (J != (mxGetM(prhs[9]))) mexErrMsgTxt("J and MUIN mismatch");
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
	ABP_voc(ALPHA, BETA, W, J, D, NN, OUTPUT, sr, ir, jc, phi, theta, mu, tw, tk, startcond);

	/* output */
	plhs[0] = mxCreateDoubleMatrix(J, D, mxREAL);
	mxSetPr(plhs[0], theta);

	plhs[1] = mxCreateDoubleMatrix(J, nzmax, mxREAL);
	mxSetPr(plhs[1], mu);
}