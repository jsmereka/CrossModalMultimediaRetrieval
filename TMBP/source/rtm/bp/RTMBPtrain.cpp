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
	int wi, di, i, j, dii, ii, jj, cii, topic, iter;
	double xi, *mu_temp, *phitot, *thetad, *gammatot, mutot, WBETA = (double) W*BETA, JALPHA = (double) J*ALPHA;

	phitot = dvec(J);
	thetad = dvec(D);
	gammatot = dvec(J);
	mu_temp = dvec(J);	

	if (startcond == 1) {
		/* start from previously saved state */
		for (di=0; di<D; di++) {
			for (i=jcwd[di]; i<jcwd[di + 1]; i++) {
				wi = (int) irwd[i];
				xi = srwd[i];
				for (j=0; j<J; j++) {
					phi[wi*J + j] += xi*mu[i*J + j]; // increment phi count matrix
					theta[di*J + j] += xi*mu[i*J + j]; // increment theta count matrix
					phitot[j] += xi*mu[i*J + j]; // increment phitot matrix
					thetad[di] += xi*mu[i*J + j];
				}
			}
		}

		/* initialize gamma using mu */
		for (di=0; di<D; di++) {
			for (ii=jcdd[di]; ii<jcdd[di + 1]; ii++) {
				dii = (int) irdd[ii];
				for (j=0; j<J; j++) {
					for (jj=0; jj<J; jj++) {
						gamma[j*J + jj] += theta[di*J + j]*theta[dii*J + jj];
						gammatot[j] += theta[di*J + j]*theta[dii*J + jj];
					}
				}
			}
		}
		for (j=0; j<J; j++) {
			for (jj=0; jj<J; jj++) {
				gamma[j*J + jj] /= gammatot[j];
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
				phi[wi*J + topic] += xi; // increment phi count matrix
				theta[di*J + topic] += xi; // increment theta count matrix
				thetad[di] += xi; // increment thetad count matrix
				phitot[topic] += xi; // increment phitot matrix
			}
		}

		/* initialize gamma using mu */
		for (di=0; di<D; di++) {
			for (ii=jcdd[di]; ii<jcdd[di + 1]; ii++) {
				dii = (int) irdd[ii];
				for (j=0; j<J; j++) {
					for (jj=0; jj<J; jj++) {
						gamma[j*J + jj] += theta[di*J + j]*theta[dii*J + jj];
						gammatot[j] += theta[di*J + j]*theta[dii*J + jj];
					}
				}
			}
		}
		for (j=0; j<J; j++) {
			for (jj=0; jj<J; jj++) {
				gamma[j*J + jj] /= gammatot[j];
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
					mu[i*J + j] = ((double) phi[wi*J + j] - (double) xi*mu[i*J + j] + (double) BETA) /
						((double) phitot[j] - (double) xi*mu[i*J + j] + (double) WBETA) *
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

		/* estimate phi, theta and phitot */
		for (i=0; i<J*W; i++) phi[i] = (double) 0;
		for (i=0; i<J*D; i++) theta[i] = (double) 0;
		for (j=0; j<J; j++) phitot[j] = (double) 0;
		for (di=0; di<D; di++) {
			for (i=jcwd[di]; i<jcwd[di + 1]; i++) {
				wi = (int) irwd[i];
				xi = srwd[i];	
				for (j=0; j<J; j++) { 
					phi[wi*J + j] += xi*mu[i*J + j]; 
					theta[di*J + j] += xi*mu[i*J + j];
					phitot[j] += xi*mu[i*J + j];					
				}
			}
		}

		/* estimate gamma */
		for (j=0; j<J*J; j++) gamma[j] = (double) 0;
		for (j=0; j<J; j++) gammatot[j] = (double) 0;
		for (di=0; di<D; di++) {
			for (ii=jcdd[di]; ii<jcdd[di + 1]; ii++) {
				dii = (int) irdd[ii];
				for (j=0; j<J; j++) {
					for (jj=0; jj<J; jj++) {
						gamma[j*J + jj] += theta[di*J + j]*theta[dii*J + jj];
						gammatot[j] += theta[di*J + j]*theta[dii*J + jj];
					}
				}
			}
		}
		for (j=0; j<J; j++) {
			for (jj=0; jj<J; jj++) {
				gamma[j*J + jj] /= gammatot[j];
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
	if (nrhs < 9) {
		mexErrMsgTxt("At least 9 input arguments required");
	} else if (nlhs < 2) {
		mexErrMsgTxt("At least 2 output arguments required");
	}

	startcond = 0;
	if (nrhs == 10) startcond = 1;

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

	J = (int) mxGetScalar(prhs[2]);
	if (J<=0) mexErrMsgTxt("Number of topics must be greater than zero");

	NN = (int) mxGetScalar(prhs[3]);
	if (NN<0) mexErrMsgTxt("Number of iterations must be positive");

	OMEGA = (double) mxGetScalar(prhs[4]);
	if (OMEGA<0 || OMEGA >1) mexErrMsgTxt("OMEGA must be in [0,1]");

	ALPHA = (double) mxGetScalar(prhs[5]);
	if (ALPHA<0) mexErrMsgTxt("ALPHA must be greater than zero");

	BETA = (double) mxGetScalar(prhs[6]);
	if (BETA<0) mexErrMsgTxt("BETA must be greater than zero");

	SEED = (int) mxGetScalar(prhs[7]);

	OUTPUT = (int) mxGetScalar(prhs[8]);

	if (startcond == 1) {
		MUIN = mxGetPr(prhs[9]);
		if (nzmaxwd != mxGetN(prhs[9])) mexErrMsgTxt("NZMAXWD and MUIN mismatch");
		if (J != mxGetM(prhs[9])) mexErrMsgTxt("J and MUIN mismatch");
	}

	/* seeding */
	seedMT( 1 + SEED * 2 ); // seeding only works on uneven numbers

	/* allocate memory */
	mu  = dvec(J*nzmaxwd);
	if (startcond == 1) {
		for (i=0; i<J*nzmaxwd; i++) mu[i] = (double) MUIN[i];   
	}

	phi = dvec(J*W);
	theta = dvec(J*D);
	gamma = dvec(J*J);

	/* run the model */
	RTMBP(OMEGA, ALPHA, BETA, W, J, D, NN, OUTPUT, mu, srwd, irwd, jcwd, nzmaxwd, irdd, jcdd, phi, theta, gamma, startcond);

	/* output */
	plhs[0] = mxCreateDoubleMatrix(J, W, mxREAL);
	mxSetPr(plhs[0], phi);

	plhs[1] = mxCreateDoubleMatrix(J, D, mxREAL );
	mxSetPr(plhs[1], theta);

	plhs[2] = mxCreateDoubleMatrix(J, J, mxREAL );
	mxSetPr(plhs[2], gamma);

	plhs[3] = mxCreateDoubleMatrix(J, nzmaxwd, mxREAL );
	mxSetPr(plhs[3], mu);
}
