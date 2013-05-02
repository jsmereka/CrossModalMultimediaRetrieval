#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mex.h>
#include "topiclib.cpp"

// Syntax
//   [ PHI , THETA , MU ] = RTMBP( WD , DD, J , N , ALPHA , BETA , SEED , OUTPUT )

// Syntax
//   [ PHI , THETA , MU ] = RTMBP( WD , DD, J , N , ALPHA , BETA , SEED , OUTPUT , MUIN )


void RTMBP(double OMEGA, double ALPHA, double BETA, int W, int J, int D, int NN, int OUTPUT, double *mu, 
	double *srwd, mwIndex *irwd, mwIndex *jcwd, int nzmaxwd,
	mwIndex *irdd, mwIndex *jcdd, double *phi, double *theta, double *gamma, int startcond) 
{
		int wi, di, i, j, dii, ii, jj, topic, iter;
		double xi, *mu_temp, *phitot, *thetad, *gammatot, mutot, WBETA = (double) W*BETA, JALPHA = (double) J*ALPHA;

		phitot = dvec(J);
		thetad = dvec(D);
		gammatot = dvec(J);
		mu_temp = dvec(J);
		
		for (i=0; i<W; i++) {
			for (j=0; j<J; j++) {
				phitot[j] = phi[i*J + j];
			}
		}

		if (startcond == 1) {
			/* start from previously saved state */
			for (di=0; di<D; di++) {
				for (i=jcwd[di]; i<jcwd[di + 1]; i++) {
					wi = (int) irwd[i];
					xi = srwd[i];
					for (j=0; j<J; j++) {
						theta[di*J + j] += xi*mu[i*J + j]; // increment theta count matrix
						thetad[di] += xi*mu[i*J + j];
					}
				}
			}
		}

		if (startcond == 0) {
			/* random initialization */
			for (di=0; di<D; di++) {
				for (i=jcwd[di]; i<jcwd[di + 1]; i++) {
					wi = (int) irwd[i];
					xi = srwd[i];
					// pick a random topic 0..J-1
					topic = (int) (J*drand());
					mu[i*J + topic] = (double) 1; // assign this word token to this topic
					theta[di*J + topic] += xi; // increment theta count matrix
					thetad[di] += xi; // increment thetad count matrix
				}
			}
		}

		for (iter=0; iter<NN; iter++) {

			if (OUTPUT >=1) {
				if ((iter % 10)==0) mexPrintf( "\tIteration %d of %d\n" , iter , NN );
				if ((iter % 10)==0) mexEvalString("drawnow;");
			}

			for (di=0; di<D; di++) {
				/* use links to calculate mu_temp */
				for (j=0; j<J; j++) mu_temp[j] = (double) 0;		
				if ((jcdd[di + 1] - jcdd[di]) > 0) {
					for (ii=jcdd[di]; ii<jcdd[di + 1]; ii++) {
						dii = (int) irdd[ii];
						for (j=0; j<J; j++) {
							for (jj=0; jj<J; jj++) {
								mu_temp[j] += gamma[j*J + jj]*((double) theta[dii*J + jj])/
									(jcdd[di + 1] - jcdd[di])/thetad[dii];
							}					
						}
					}
				}
				/* iterate all tokens */
				for (i=jcwd[di]; i<jcwd[di + 1]; i++) {				
					wi = (int) irwd[i];
					xi = srwd[i];
					/* calculate the posterior probability */
					mutot = (double) 0;
					for (j=0; j<J; j++) {
						mu[i*J + j] = ((double) phi[wi*J + j] + (double) BETA) /
							((double) phitot[j] + (double) WBETA) *
							((1 - OMEGA) * ((double) theta[di*J + j] - (double) xi*mu[i*J + j] + (double) ALPHA) /
							((double) thetad[di] - (double) xi + (double) JALPHA) 
							+ OMEGA * (double) mu_temp[j]);
						mutot += mu[i*J + j];
					}
					for (j=0; j<J; j++) {
						mu[i*J + j] /= mutot;
					}
				}
			}

			/* clear and update theta */
			for (i=0; i<J*D; i++) theta[i] = (double) 0;
			for (di=0; di<D; di++) {
				for (i=jcwd[di]; i<jcwd[di + 1]; i++) {
					wi = (int) irwd[i];
					xi = srwd[i];	
					for (j=0; j<J; j++) { 
						theta[di*J + j] += xi*mu[i*J + j];				
					}
				}
			}
		}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	double *MUIN, *phi, *theta, *gamma, *mu, *srwd, *srdd;
	double ALPHA, BETA, OMEGA;
	mwIndex *irwd, *jcwd, *irdd, *jcdd;
	int W, J, D, NN, SEED, OUTPUT, nzmaxwd, nzmaxdd, i, j, startcond;

	/* Check for proper number of arguments. */
	if (nrhs < 10) {
		mexErrMsgTxt("At least 10 input arguments required");
	} else if (nlhs < 1) {
		mexErrMsgTxt("At least 1 output arguments required");
	}

	startcond = 0;
	if (nrhs == 11) startcond = 1;

	/* dealing with sparse array WD */
	if (mxIsDouble(prhs[0]) != 1) mexErrMsgTxt("WD must be a double precision matrix");
	srwd = mxGetPr(prhs[0]);
	irwd = mxGetIr(prhs[0]);
	jcwd = mxGetJc(prhs[0]);
	nzmaxwd = (int) mxGetNzmax(prhs[0]);
	W = (int) mxGetM(prhs[0]);
	D = (int) mxGetN(prhs[0]);

	/* dealing with sparse array DD */
	if (mxIsDouble(prhs[1]) != 1) mexErrMsgTxt("WD must be a double precision matrix");
	srdd = mxGetPr(prhs[1]);
	irdd = mxGetIr(prhs[1]);
	jcdd = mxGetJc(prhs[1]);
	nzmaxdd = (int) mxGetNzmax(prhs[1]);
	if ( mxGetM(prhs[1]) != D) mexErrMsgTxt("WD and DD mismatch"); 

	/* check links between documents */
	for (i=0; i<nzmaxdd; i++) {
		if ((int) srdd[i] != 1) mexErrMsgTxt("Entries in DD matrix can only be 0 or 1");
	}

	phi = mxGetPr(prhs[2]);
	J = (int) mxGetM(prhs[2]);
	if (J<=0) mexErrMsgTxt("Number of topics must be greater than zero");
	if ((int) mxGetN(prhs[2]) != W) mexErrMsgTxt("Vocabulary mismatches");

	gamma = mxGetPr(prhs[3]);
	if (J != mxGetM(prhs[3])) mexErrMsgTxt("gamma is not JxJ matrix");
	if (J != mxGetN(prhs[3])) mexErrMsgTxt("gamma is not JxJ matrix");

	NN = (int) mxGetScalar(prhs[4]);
	if (NN<0) mexErrMsgTxt("Number of iterations must be positive");

	OMEGA = (double) mxGetScalar(prhs[5]);
	if (OMEGA<0 || OMEGA >1) mexErrMsgTxt("OMEGA must be in [0,1]");

	ALPHA = (double) mxGetScalar(prhs[6]);
	if (ALPHA<0) mexErrMsgTxt("ALPHA must be greater than zero");

	BETA = (double) mxGetScalar(prhs[7]);
	if (BETA<0) mexErrMsgTxt("BETA must be greater than zero");

	SEED = (int) mxGetScalar(prhs[8]);

	OUTPUT = (int) mxGetScalar(prhs[9]);

	if (startcond == 1) {
		MUIN = mxGetPr(prhs[10]);
		if (nzmaxwd != mxGetN(prhs[10])) mexErrMsgTxt("NZMAXWD and MUIN mismatch");
		if (J != mxGetM(prhs[10])) mexErrMsgTxt("J and MUIN mismatch");
	}

	/* seeding */
	seedMT(1 + SEED*2); // seeding only works on uneven numbers

	/* allocate memory */
	mu = dvec(J*nzmaxwd);
	if (startcond == 1) {
		for (i=0; i<J*nzmaxwd; i++) mu[i] = (double) MUIN[i];   
	}

	theta = dvec(J*D);

	/* run the model */
	RTMBP(OMEGA, ALPHA, BETA, W, J, D, NN, OUTPUT, mu, srwd, irwd, jcwd, nzmaxwd, irdd, jcdd, phi, theta, gamma, startcond);

	/* output */
	plhs[0] = mxCreateDoubleMatrix(J, D, mxREAL );
	mxSetPr(plhs[0], theta);

	plhs[1] = mxCreateDoubleMatrix(J, nzmaxwd, mxREAL );
	mxSetPr(plhs[1], mu);
}
