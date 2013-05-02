#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mex.h>
#include "topiclib.cpp"

// Syntax
//   [ PHI , THETA , MU ] = ABP_doc( WD , J , TD , TK , N , ALPHA , BETA , SEED , OUTPUT )

// Syntax
//   [ PHI , THETA , MU ] = ABP_doc( WD , J , TD , TK , N , ALPHA , BETA , SEED , OUTPUT , MUIN )

/* ABP_doc algorithm */
void ABP_doc(double ALPHA, double BETA, int W, int J, int D, int NN, int OUTPUT, 
	double *sr, mwIndex *ir, mwIndex *jc, double *phi, double *theta, double *mu, double td, double tk, int startcond) 
{
	int wi, di, i, j, k, topic, iter, ii;
	int *ind_rk, *ind_rd;
	double mutot, totprob, xi, xitot, perp;
	double JALPHA = (double) (J*ALPHA), WBETA = (double) (W*BETA);
	double *phitot, *thetatot, *rk, *rd, *munew;

	phitot = dvec(J);
	thetatot = dvec(D);
	ind_rk = ivec(J*D);
	ind_rd = ivec(D);
	munew = dvec(J);
	rk = dvec(J*D);
	rd = dvec(D);

	if (startcond == 1) {
		/* start from previously saved state */
		for (di=0; di<D; di++) {
			for (i=jc[di]; i<jc[di + 1]; i++) {
				wi = (int) ir[i];
				xi = sr[i];
				xitot += xi;
				thetatot[di] += xi;
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
				thetatot[di] += xi;
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
								(theta[di*J + j] + ALPHA)/(thetatot[di] + JALPHA);
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
			for (di=0; di<D; di++) {
				for (i=jc[di]; i<jc[di + 1]; i++) {
					wi = (int) ir[i];
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
						rk[di*J + j] += xi*fabs(munew[j] - mu[i*J + j]);
						rd[di] += xi*fabs(munew[j] - mu[i*J + j]);
						mu[i*J + j] = munew[j];
						phi[wi*J + j] += xi*mu[i*J + j];
						phitot[j] += xi*mu[i*J + j];
						theta[di*J + j] += xi*mu[i*J + j];
					}
				}
				dsort(J, rk + di*J, -1, ind_rk + di*J);
			}
			dsort(D, rd, -1, ind_rd);

		} else { /* iteration > 0 */
			for (ii=0; ii < (int) (td*D); ii++) {
				di = (int) ind_rd[ii];
				for (j=0; j < (int) (J*tk); j++) {				
					k = (int) ind_rk[di*J + j];
					rd[di] -= rk[di*J + k];
					rk[di*J + k] = 0.0;
				}
				for (i=jc[di]; i<jc[di + 1]; i++) {
					wi = (int) ir[i];
					xi = sr[i];
					for (j=0, mutot=0.0, totprob=0.0; j < (int) (J*tk); j++) {
						k = (int) ind_rk[di*J + j];
						phi[wi*J + k] -= xi*mu[i*J + k];
						phitot[k] -= xi*mu[i*J + k];
						theta[di*J + k] -= xi*mu[i*J + k];	
						totprob += mu[i*J + k];
						munew[k] = (phi[wi*J + k] + BETA)/(phitot[k] + WBETA)*(theta[di*J + k] + ALPHA);
						mutot += munew[k];
					}
					for (j=0; j < (int) (J*tk); j++) {
						k = (int) ind_rk[di*J + j];
						munew[k] /= mutot;
						munew[k] *= totprob;
						rk[di*J + k] += xi*fabs(munew[k] - mu[i*J + k]);
						mu[i*J + k] = munew[k];
						phi[wi*J + k] += xi*mu[i*J + k];
						phitot[k] += xi*mu[i*J + k];
						theta[di*J + k] += xi*mu[i*J + k];
					}
				}
				for (j=0; j < (int) (J*tk); j++) {
					k = (int) ind_rk[di*J + j];
					rd[di] += rk[di*J + k];
				}
				insertionsort(J, rk + di*J, ind_rk + di*J);
			}
			insertionsort(D, rd, ind_rd);

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
	double ALPHA, BETA, td, tk;
	mwIndex *ir, *jc;
	int W, J, D, NN, SEED, OUTPUT, nzmax, i, startcond;

	/* Check for proper number of arguments. */
	if (nrhs < 9) {
		mexErrMsgTxt("At least 9 input arguments required");
	} else if (nlhs < 2) {
		mexErrMsgTxt("At least 2 output arguments required");
	}

	startcond = 0;
	if (nrhs == 10) startcond = 1;

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

	td = (double) mxGetScalar(prhs[2]);
	if (td<0 || td>1) mexErrMsgTxt("td should be between 0 and 1.");

	tk = (double) mxGetScalar(prhs[3]);
	if (tk<0 || tk>1) mexErrMsgTxt("tk should be between 0 and 1.");

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
		if (nzmax != (mxGetN(prhs[9]))) mexErrMsgTxt("WD and MUIN mismatch");
		if (J != (mxGetM(prhs[9]))) mexErrMsgTxt("J and MUIN mismatch");
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
	ABP_doc(ALPHA, BETA, W, J, D, NN, OUTPUT, sr, ir, jc, phi, theta, mu, td, tk, startcond);

	/* output */
	plhs[0] = mxCreateDoubleMatrix(J, W, mxREAL);
	mxSetPr(plhs[0], phi);

	plhs[1] = mxCreateDoubleMatrix(J, D, mxREAL);
	mxSetPr(plhs[1], theta);

	plhs[2] = mxCreateDoubleMatrix(J, nzmax, mxREAL);
	mxSetPr(plhs[2], mu);
}

