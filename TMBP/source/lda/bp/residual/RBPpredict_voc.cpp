#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mex.h>
#include "topiclib.cpp"

// Syntax
//   [ PHI , THETA , MU ] = RBP_voc( WD , J , N , ALPHA , BETA , SEED , OUTPUT )

// Syntax
//   [ PHI , THETA , MU ] = RBP_voc( WD , J , N , ALPHA , BETA , SEED , OUTPUT , MUIN )

/* RBP_voc algorithm */
void RBP_voc(double ALPHA, double BETA, int W, int J, int D, int NN, int OUTPUT, int nzmax, double *sr, mwIndex *ir, mwIndex *jc,
	double *phi, double *theta, double *mu, int startcond) 
{
	int wi, di, i, j, topic, iter, *order, ii;
	double *phitot, mutot, *thetad, *r, *munew, JALPHA = (double) (J*ALPHA);
	double WBETA = (double) (W*BETA), xi, xitot, perp, tmp;

	phitot = dvec(J);

	for (wi=0; wi<W; wi++) {
		for (j=0; j<J; j++) {
			phitot[j] += phi[wi*J + j];
		}
	}

	munew = dvec(J);
	thetad = dvec(D);
	order = ivec(W);
	r = dvec(W);

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

	/* Initial order */
	for (i=0; i<W; i++) order[i] = i;

	for (iter=0; iter<NN; iter++) {

		if (OUTPUT >= 1) {
			if (((iter % 10)==0) && (iter != 0)) {
				/* calculate perplexity */
				perp = 0.0;
				for (wi=0; wi<W; wi++) {
					for (i=jc[wi]; i<jc[wi + 1]; i++) {
						di = (int) ir[i];
						xi = (double) sr[i];
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

		/* passing message mu */
		for (ii=0; ii<W; ii++) {
			wi = (int) order[ii];
			r[wi] = 0.0;
			for (i=jc[wi]; i<jc[wi + 1]; i++) {
				di = (int) ir[i];
				xi = sr[i];
				mutot = (double) 0;
				for (j=0; j<J; j++) {
					theta[di*J + j] -= (double) xi*mu[i*J + j];
					munew[j] = ((double) phi[wi*J + j] + (double) BETA)/
						((double) phitot[j] + (double) WBETA)*
						((double) theta[di*J + j] + (double) ALPHA);
					mutot += munew[j];
				}
				for (j=0; j<J; j++) {
					munew[j] /= mutot;
					r[wi] += xi*fabs(munew[j] - mu[i*J + j]);
					mu[i*J + j] = (double) munew[j];
					theta[di*J + j] += (double) xi*mu[i*J + j];
				}
			}
		}

		/* Schedule order based on residues */
		if (iter == 0) {
			dsort(W, r, -1, order);
		} else {
			insertionsort(W, r, order);
		}
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	double *mu, *MUIN, *phi, *theta, *sr;
	double ALPHA, BETA;
	mwIndex *ir, *jc;
	int W, J, D, NN, SEED, OUTPUT, nzmax, i, j, wi, di, startcond;

	/* Check for proper number of arguments. */
	if (nrhs < 7) {
		mexErrMsgTxt("At least 7 input arguments required");
	} else if (nlhs < 1) {
		mexErrMsgTxt("At least 1 output arguments required");
	}

	startcond = 0;
	if (nrhs == 8) startcond = 1;

	/* read sparse array WD */
	if (mxIsDouble(prhs[0]) != 1) mexErrMsgTxt("WD must be a double precision matrix");
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
	mu = dvec(J*nzmax);
	if (startcond == 1) {
		for (i=0; i<J*nzmax; i++) mu[i] = (double) MUIN[i];   
	}
	theta = dvec(J*D);

	/* run the learning algorithm */
	RBP_voc(ALPHA, BETA, W, J, D, NN, OUTPUT, nzmax, sr, ir, jc, phi, theta, mu, startcond);

	/* output */
	plhs[0] = mxCreateDoubleMatrix(J, D, mxREAL );
	mxSetPr(plhs[0], theta);

	plhs[1] = mxCreateDoubleMatrix(J, nzmax, mxREAL );
	mxSetPr(plhs[1], mu);
}

