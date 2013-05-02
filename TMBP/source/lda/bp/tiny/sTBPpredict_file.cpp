#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mex.h>
#include "topiclib.cpp"
#define MAX 50000 // maximum length of the line is 50000 characters

// Syntax
// [THETA] = sTBP(filename, PHI, N, ALPHA, BETA, SEED, OUTPUT)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
	const mxArray *prhs[]) 
{
	double *mu, *phi, *theta, *phitot, *theta2, mutot, WBETA, JALPHA, ALPHA, BETA, perp;
	int W1 = 0, W, J, D = 0, NN, SEED, OUTPUT, i, j, buflen, nnz, iter, di, wi, xi;
	double *thetad, xitot = 0;
	char *filename, *pch, mystring[MAX];
	FILE *doc;

	/* Check for proper number of arguments. */
	if (nrhs < 7) {
		mexErrMsgTxt("At least 7 input arguments required");
	} else if (nlhs < 1) {
		mexErrMsgTxt("1 output arguments required");
	}

	// open document file
	buflen = (int) (1 + mxGetM(prhs[0])*mxGetN(prhs[0]));
	filename = (char *) mxCalloc(buflen, sizeof(char));
	mxGetString(prhs[0], filename, buflen);
	doc = fopen(filename, "r"); 
	if (doc == NULL) mexErrMsgTxt("File cannot be found.");

	phi = (double *) mxGetPr(prhs[1]);
	J = (int) mxGetM(prhs[1]);
	if (J<=0) mexErrMsgTxt("Number of topics must be greater than zero");
	W = (int) mxGetN(prhs[1]);
	if (W<=0) mexErrMsgTxt("Vocabulary must be greater than zero");

	NN = (int) mxGetScalar(prhs[2]);
	if (NN<0) mexErrMsgTxt("Number of iterations must be positive");

	ALPHA = (double) mxGetScalar(prhs[3]);
	if (ALPHA<=0) mexErrMsgTxt("ALPHA must be greater than zero");

	BETA = (double) mxGetScalar(prhs[4]);
	if (BETA<=0) mexErrMsgTxt("BETA must be greater than zero");

	SEED = (int) mxGetScalar(prhs[5]);

	OUTPUT = (int) mxGetScalar(prhs[6]);

	// seeding
	seedMT(1 + SEED*2); // seeding only works on uneven numbers 

	// read the file header
	fgets(mystring, sizeof(mystring), doc);

	// Unix file contains a special space at each line of the file
	pch = strtok(mystring, " ");
	if ((pch != NULL) && (atoi(pch) != 0)){
		D = atoi(pch);
		mexPrintf( "#document: %d \n", D);
	} else {
		mexErrMsgTxt("File header error.");
	}

	pch = strtok(NULL, " ");
	if ((pch != NULL) && (atoi(pch) != 0)){
		W1 = atoi(pch);
		mexPrintf( "#Vocabulary: %d \n", W);
	} else {
		mexErrMsgTxt("File header error.");
	}

	if (W1 != W) mexErrMsgTxt("Vocabulary mismatches phi.");

	pch = strtok(NULL, " ");
	if ((pch != NULL) && (atoi(pch) != 0)){
		nnz = atoi(pch);
		mexPrintf( "#token: %d \n", nnz);
	} else {
		mexErrMsgTxt("File header error.");
	}
	if (nnz == 0) mexErrMsgTxt("Empty file.");

	/* run the model */

	// allocate memory
	theta = dvec(J*D);
	mu  = dvec(J);
	theta2 = dvec(J*D);
	thetad = dvec(D);

	// random initialize theta
	for (i=0; i<J*D; i++) theta[i] = drand();

	// calculate phitot
	phitot = dvec(J);
	for (j=0; j<J; j++) {
		for (i=0; i<W; i++) {
			phitot[j] += phi[i*J + j];
		}
	}

	// message passing
	WBETA = (double) (W*BETA);
	JALPHA = (double) (J*ALPHA);

	for (iter=0; iter<NN; iter++) {

		if (OUTPUT >= 1) {
			if (((iter % 10)==0) && (iter != 0)) {
				/* calculate perplexity */
				perp = (double) 0;
				di = (int) 0;
				while (fgets(mystring , sizeof(mystring) , doc) != NULL) {
					pch = strtok(mystring, " ");
					if ((pch != NULL) && (atoi(pch) != 0)) {
						wi = atoi(pch) - 1;
						pch = strtok(NULL, " ");
						xi = atof(pch);
						mutot = (double) 0;
						for (j=0; j<J; j++) {
							mutot += ((double) phi[wi*J + j] + (double) BETA)/
								((double) phitot[j] + (double) WBETA)*
								((double) theta[di*J + j] + (double) ALPHA)/
								((double) thetad[di] + (double) JALPHA);;
						}
						perp -= (log(mutot)*xi);
					}
					while ((pch != NULL) && (atoi(pch) != 0)) {
						pch = strtok(NULL, " ");
						if ((pch != NULL) && (atoi(pch) != 0)) {
							wi = atoi(pch)-1;
							pch = strtok(NULL, " ");
							xi = atof(pch);
							mutot = (double) 0;
							for (j=0; j<J; j++) {
								mutot += ((double) phi[wi*J + j] + (double) BETA)/
									((double) phitot[j] + (double) WBETA)*
									((double) theta[di*J + j] + (double) ALPHA)/
									((double) thetad[di] + (double) JALPHA);;
							}
							perp -= (log(mutot)*xi);
						}
					}
					di++; // increment doc index
				}
				mexPrintf("\tIteration %d of %d:\t%f\n", iter, NN, exp(perp/xitot));
				if ((iter % 10)==0) mexEvalString("drawnow;");

				// return and skip the file header
				rewind(doc);
				fgets(mystring, sizeof(mystring), doc);
			}
		}

		// clear theta2
		for (i=0; i<J*D; i++) theta2[i] = (double) 0;

		// scan the entire file
		di = (int) 0;
		while (fgets(mystring, sizeof(mystring), doc) != NULL) {
			pch = strtok(mystring, " ");
			if ((pch != NULL) && (atoi(pch) != 0)) {
				wi = atoi(pch)-1;
				pch = strtok(NULL, " ");
				xi = atof(pch);
				if (iter == 0) {
					thetad[di] += xi;
					xitot += xi;
				}
				mutot = (double) 0;
				for (j = 0; j < J; j++) {
					mu[j] = (phi[wi*J+j] + BETA)/(phitot[j] + WBETA)*(theta[di*J + j] + ALPHA);
					mutot += mu[j];
				}
				for (j = 0; j < J; j++) {  
					theta2[di*J + j] += (mu[j]/mutot)*xi;
				}
			}
			while ((pch != NULL) && (atoi(pch) != 0)) {
				pch = strtok(NULL, " ");
				if ((pch != NULL) && (atoi(pch) != 0)) {
					wi = atoi(pch)-1;
					pch = strtok(NULL, " ");
					xi = atof(pch);
					if (iter == 0) {
						thetad[di] += xi;
						xitot += xi;
					}
					mutot = (double) 0;
					for (j=0; j<J; j++) {
						mu[j] = (phi[wi*J + j] + BETA)/(phitot[j] + WBETA)*(theta[di*J + j] + ALPHA);
						mutot += mu[j];
					}
					for (j=0; j<J; j++) { 
						theta2[di*J + j] += (mu[j]/mutot)*xi;
					}
				}
			}
			di++; // increment doc index
		}

		// return and skip the file header
		rewind(doc);
		fgets(mystring, sizeof(mystring), doc);

		// theta
		for (i=0; i<J*D; i++) theta[i] = theta2[i];
	}

	/* output */
	plhs[0] = mxCreateDoubleMatrix(J, D, mxREAL);
	mxSetPr(plhs[0], theta);
}
