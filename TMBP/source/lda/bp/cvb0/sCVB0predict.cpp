#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mex.h>
#include "topiclib.cpp"

// Syntax
//   [ phi , theta , mu ] = sCVB0( WD , J , N , ALPHA , BETA , SEED , OUTPUT )

// Syntax
//   [ phi , theta , mu ] = sCVB0( WD , J , N , ALPHA , BETA , SEED , OUTPUT , ZIN )


void sCVB0(double ALPHA, double BETA, int W, int J, int D, int NN, int OUTPUT, int ntokens, int *d, int *w, 
	double *phi, double *theta, double *mu, int startcond)
{
	int wi, di, i, ii, j, topic, rp, temp, iter;
	double totprob, *thetad, *phitot, xitot = 0.0, JALPHA = (double) (J*ALPHA), WBETA = (double) (W*BETA), r, max, perp = 0.0;

	thetad = dvec(D);
	phitot = dvec(J);

	for (wi=0; wi<W; wi++) {
		for (j=0; j<J; j++) {
			phitot[j] += (double) phi[wi*J + j];
		}
	}

	if (startcond == 1) {
		/* start from previously saved state */
		for (i=0; i<ntokens; i++) {
			wi = w[i];
			di = d[i];
			xitot++;
			thetad[di]++;
			for (j=0; j<J; j++) {
				theta[di*J + topic] += mu[i*J + j]; // increment theta count matrix
			}
		}
	}

	if (startcond == 0) {
		/* random initialization */
		for (i=0; i<ntokens; i++) {
			wi = w[i];
			di = d[i];
			xitot++;
			thetad[di]++;
			// pick a random topic 0..J-1
			topic = (int) (J*drand());
			theta[di*J + topic]++; // increment theta count matrix
			mu[i*J + topic] = (double) 1;
		}
	}

	for (iter=0; iter<NN; iter++) {

		if (((iter % 10)==0) && (iter != 0)) {
			/* calculate perplexity */
			perp = (double) 0;
			for (i=0; i<ntokens; i++) {
				wi = w[i]; // current word index
				di = d[i]; // current document index  
				totprob = (double) 0;
				for (j=0; j<J; j++) {
					totprob += ((double) phi[wi*J + j] + (double) BETA)/
						((double) phitot[j]+ (double) WBETA)*
						((double) theta[di*J + j] + (double) ALPHA)/
						((double) thetad[di] + (double) JALPHA);
				}
				perp -= log(totprob);
			}
			mexPrintf("\tIteration %d of %d:\t%f\n", iter, NN, exp(perp/xitot));
			if ((iter % 10)==0) mexEvalString("drawnow;");
		}

		for (i=0; i<ntokens; i++) {
			wi = w[i]; // current word index
			di = d[i]; // current document index  

			totprob = (double) 0;
			for (j=0; j<J; j++) {
				mu[i*J + j] = ((double) phi[ wi*J+j ] + (double) BETA)/( (double) phitot[j] + (double) WBETA)*
					((double) theta[ di*J+j ] - (double) mu[i*J + j] + (double) ALPHA);
				totprob += mu[i*J + j];
			}         
			for (j=0; j<J; j++) {
				mu[i*J + j] /= totprob;
			}
		}

		for (i=0; i<J*D; i++) theta[i] = (double) 0;

		for (i=0; i<ntokens; i++) {
			di = d[i]; // current document index 
			for (j=0; j<J; j++) {
				theta[di*J + j] += mu[i*J + j];
			}
		}
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	double *mu, *MUIN, *sr, *phi, *theta;
	double ALPHA, BETA;
	mwIndex *ir, *jc;
	int *z, *d, *w;
	int W, J, D, NN, SEED, OUTPUT, nzmax, ntokens;
	int k, i, j, wi, di, xi, startcond;

	/* Check for proper number of arguments. */
	if (nrhs < 7) {
		mexErrMsgTxt("At least 7 input arguments required");
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

	/* get the number of tokens */
	ntokens = (int) 0;
	for  (i=0; i<nzmax; i++) ntokens += (int) sr[i];
	if (ntokens == 0) mexErrMsgTxt("word vector is empty"); 

	/* get phi */
	phi = mxGetPr(prhs[1]);
	J = (int) mxGetM(prhs[1]);
	if (J<=0) mexErrMsgTxt("Number of topics must be greater than zero");
	if ((int) mxGetN(prhs[1]) != W) mexErrMsgTxt("Vocabulary mismatches");

	NN = (int) mxGetScalar(prhs[2]);
	if (NN<0) mexErrMsgTxt("Number of iterations must be positive");

	ALPHA = (double) mxGetScalar(prhs[3]);
	if (ALPHA<=0) mexErrMsgTxt("ALPHA must be greater than zero");

	BETA = (double) mxGetScalar(prhs[4]);
	if (BETA<=0) mexErrMsgTxt("BETA must be greater than zero");

	SEED = (int) mxGetScalar(prhs[5]);

	OUTPUT = (int) mxGetScalar(prhs[6]);

	if (startcond == 1) {
		MUIN = mxGetPr(prhs[7]);
		if (J*ntokens != (mxGetM(prhs[7])*mxGetN(prhs[7]))) mexErrMsgTxt("Word and ZIN vectors should have same number of entries");
	}

	/* seeding */
	seedMT(1 + SEED*2); // seeding only works on uneven numbers


	/* allocate memory */
	mu = dvec(J*ntokens);
	if (startcond == 1) {
		for (i=0; i<J*ntokens; i++) mu[i] = (double) MUIN[i];   
	}

	/* copy word and document indices */
	d = ivec(ntokens);
	w = ivec(ntokens);
	k = (int) 0;
	for (di=0; di<D; di++) {
		for (i=jc[di]; i<jc[di+1]; i++) {
			wi = (int) ir[i];
			xi = (int) sr[i];
			for (j=0; j<xi; j++) {
				d[k] = di;
				w[k] = wi;
				k++;
			}
		}
	}
	if (ntokens != k) mexErrMsgTxt("Fail to read data");

	/* allocate memory to parameters */
	theta = dvec(J*D);

	/* run the learning algorithm */
	sCVB0(ALPHA, BETA, W, J, D, NN, OUTPUT, ntokens, d, w, phi, theta, mu, startcond);

	/* output */
	plhs[0] = mxCreateDoubleMatrix(J, D, mxREAL);
	mxSetPr(plhs[0], theta);

	plhs[1] = mxCreateDoubleMatrix(J, ntokens, mxREAL);
	mxSetPr(plhs[1], mu);
}
