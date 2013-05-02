#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mex.h>
#include "topiclib.cpp"

// Syntax
//   [ phi , theta , Z ] = FGS( WD , J , N , ALPHA , BETA , SEED , OUTPUT )

// Syntax
//   [ phi , theta , Z ] = FGS( WD , J , N , ALPHA , BETA , SEED , OUTPUT , ZIN )

/* fast GS algorithm */
void FGS(double ALPHA, double BETA, int W, int J, int D, int NN, int OUTPUT, int ntokens, 
	int *z, int *d, int *w, int *phi, int *theta, int startcond) 
{
	int wi, di, i, ii, j, k, jj, topic_new, topic, rp, temp, iter;
	int *indx, *rindx, *indx_z, *rindx_z, *thetad, *phitot, *order, *d_last_z, *w_last_z;
	double mutot, xitot, u, r, max, perp, phitotnorm, thetanorm2, phinorm2, zk, zk_old;
	double *totprob, *thetanorm, *phinorm, *mu;
	double JALPHA = (double) (J*ALPHA), WBETA = (double) (W*BETA);

	thetad = ivec(D);
	phitot = ivec(J);
	thetanorm = dvec(D);
	phinorm = dvec(W);
	indx = ivec(J*D);
	rindx = ivec(J*D);
	indx_z = ivec(J);
	rindx_z = ivec(J);
	totprob = dvec(J);
	mu = dvec(J);
	d_last_z = ivec(D);
	w_last_z = ivec(W);
	order = ivec(ntokens);

	if (startcond == 1) {
		/* start from previously saved state */
		for (i=0; i<ntokens; i++) {
			wi = w[i];
			di = d[i];
			topic = z[i];
			phi[wi*J + topic]++; // increment phi count matrix
			theta[di*J + topic]++; // increment theta count matrix
			phitot[topic]++; // increment phitot matrix
			thetad[di]++;
			xitot++;
		}
	}

	if (startcond == 0) {
		/* random initialization */
		for (i=0; i<ntokens; i++) {
			wi = w[i];
			di = d[i];
			// pick a random topic 0..J-1
			topic = (int) (J*drand());
			z[i] = topic; // assign this word token to this topic
			phi[wi*J + topic]++; // increment phi count matrix
			theta[di*J + topic]++; // increment theta count matrix
			phitot[topic]++; // increment phitot matrix
			thetad[di]++;
			xitot++;
		}
	}

	/* Determine the random order */
	for (i=0; i<ntokens; i++) order[i] = i; // fill with increasing series
	for (i=0; i<(ntokens-1); i++) {
		// pick a random integer between i and nw
		rp = i + (int) ((ntokens-i)*drand());
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
				for (i=0; i<ntokens; i++) {
					wi = w[i]; // current word index
					di = d[i]; // current document index  
					mutot = 0.0;
					for (j=0; j<J; j++) {
						mutot += ((double) phi[wi*J + j] + (double) BETA)/
							((double) phitot[j] + (double) WBETA)*
							((double) theta[di*J + j] + (double) ALPHA)/
							((double) thetad[di] + (double) JALPHA);
					}
					perp -= log(mutot);
				}
				mexPrintf("\tIteration %d of %d:\t%f\n", iter, NN, exp(perp/xitot));
				if ((iter % 10)==0) mexEvalString("drawnow;");
			}
		}

		for (ii=0; ii<ntokens; ii++) {

			i = order[ii]; // current word token to assess
			wi = w[i]; // current word index
			di = d[i]; // current document index  

			topic = z[i]; // current topic assignment to word token
			phitot[topic]--;  // substract this from counts
			phi[wi*J + topic]--;
			theta[di*J + topic]--;

			if (iter == 0) {
				// setup phitotnorm
				if (ii == 0) {
					isort(J, phitot, -1, indx_z);
					isort(J, indx_z, 1, rindx_z);
				} else {
					if (topic != topic_new) updatesort(J, phitot, indx_z, rindx_z, topic_new, topic);
				}
				phitotnorm = phitot[indx_z[J - 1]] + WBETA;

				// sort theta
				if (ii == ntokens-1 || d[order[ii+1]] != d[order[ii]]) {
					isort(J, theta + di*J, -1, indx + di*J);
					isort(J, indx + di*J, 1, rindx + di*J);
				}

				// setup norms
				thetanorm[di] = 0;
				phinorm[wi] = 0;
				for (j=0; j<J; j++) {
					thetanorm[di] += SQR((double) theta[di*J + j] + (double) ALPHA);
					phinorm[wi] += SQR((double) phi[wi*J + j] + (double) BETA);
				}

				// usual sample for iter == 0
				mutot = (double) 0;
				for (j=0; j<J; j++) {
					mu[j] = ((double) phi[wi*J + j] + (double) BETA)/
						((double) phitot[j]+ (double) WBETA)*
						((double) theta[di*J + j] + (double) ALPHA);
					mutot += mu[j];
				}

				// sample a topic from the distribution
				r = mutot*drand();
				max = mu[0];
				topic_new = 0;
				while (r > max) {
					topic_new++;
					max += mu[topic_new];
				}
			} else {

				if (topic_new != topic) {
					updatesort(J, phitot, indx_z, rindx_z, topic_new, topic);
					phitotnorm = phitot[indx_z[J - 1]] + WBETA;
				}

				if (d_last_z[di] != topic) {
					thetanorm[di] += 2*(theta[di*J + d_last_z[di]] - theta[di*J + topic] - 1);
					updatesort(J, theta + di*J, indx + di*J, rindx + di*J, d_last_z[di], topic);
				}
				thetanorm2 = thetanorm[di];

				if (w_last_z[wi] != topic) {
					phinorm[wi] += 2*(phi[wi*J + w_last_z[wi]] - phi[wi*J + topic] - 1);
				}
				phinorm2 = phinorm[wi];

				u = drand();
				for (j=0; j<J; j++) {
					k = (int) indx[di*J + j];	
					mu[j] = ((double) phi[wi*J + k] + (double) BETA)/
						((double) phitot[k]+ (double) WBETA)*
						((double) theta[di*J + k] + (double) ALPHA);
					if (j == 0) {
						totprob[j] = mu[j];
					} else {
						totprob[j] = totprob[j - 1] + mu[j];
					}
					thetanorm2 -= SQR((double) theta[di*J + k] + (double) ALPHA); // update theta norm
					phinorm2 -= SQR((double) phi[wi*J + k] + (double) BETA); // update phi norm
					if (thetanorm2 < 0) thetanorm2 = 0;
					if (phinorm2 < 0) phinorm2 = 0;
					zk_old = zk;
					zk = totprob[j] + sqrt(thetanorm2*phinorm2)/phitotnorm;		

					// sample a topic from the distribution
					r = u*zk;
					if (totprob[j] < r) continue;
					else {
						if ((j == 0) || (r > totprob[j - 1])) {
							topic_new = indx[di*J + j];
							break;
						} else {
							u = (u*zk_old - totprob[j - 1])*zk/(zk_old - zk);
							for (jj = 0; jj<j; jj++) {
								if (totprob[jj] >= u) {
									topic_new = indx[di*J + jj];
									break;
								}
							} break;
						}
					}
				}
			}

			z[i] = topic_new; // assign current word token i to topic j
			phi[wi*J + topic_new]++; // and update counts
			theta[di*J + topic_new]++;
			phitot[topic_new]++;

			d_last_z[di] = topic_new;
			w_last_z[wi] = topic_new;
		}
		if (iter > NN - 10) {
			mexPrintf("\tIter %d  Sec %.2f\n", iter, etime());
			mexEvalString("drawnow;");
		}
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	double *srphi, *srtheta, *Z, *ZIN, *sr;
	double ALPHA, BETA;
	mwIndex *irphi, *jcphi, *irtheta, *jctheta, *ir, *jc;
	int *z, *d, *w, *phi, *theta;
	int W, J, D, NN, SEED, OUTPUT, nzmax, nzmaxphi, nzmaxtheta, ntokens;
	int k, i, j, c, n, wi, di, xi, startcond;

	/* Check for proper number of arguments. */
	if (nrhs < 7) {
		mexErrMsgTxt("At least 7 input arguments required");
	} else if (nlhs < 2) {
		mexErrMsgTxt("At least 2 output arguments required");
	}

	startcond = 0;
	if (nrhs == 8) startcond = 1;

	/* read sparse array WD */
	if (mxIsDouble(prhs[0]) != 1) mexErrMsgTxt("WD must be a double precision matrix");
	sr = mxGetPr(prhs[0]);
	ir = mxGetIr(prhs[0]);
	jc = mxGetJc(prhs[0]);
	nzmax = (int) mxGetNzmax(prhs[0]);
	W = (int) mxGetM(prhs[0]);
	D = (int) mxGetN(prhs[0]);
	ntokens = (int) 0;
	for  (i=0; i<nzmax; i++) ntokens += (int) sr[i];
	if (ntokens == 0) mexErrMsgTxt("word vector is empty"); 

	J = (int) mxGetScalar(prhs[1]);
	if (J<=0) mexErrMsgTxt("Number of topics must be greater than zero");

	NN = (int) mxGetScalar(prhs[2]);
	if (NN<0) mexErrMsgTxt("Number of iterations must be positive");

	ALPHA = (double) mxGetScalar(prhs[3]);
	if (ALPHA<=0) mexErrMsgTxt("ALPHA must be greater than zero");

	BETA = (double) mxGetScalar(prhs[4]);
	if (BETA<=0) mexErrMsgTxt("BETA must be greater than zero");

	SEED = (int) mxGetScalar(prhs[5]);

	OUTPUT = (int) mxGetScalar(prhs[6]);

	if (startcond == 1) {
		ZIN = mxGetPr(prhs[7]);
		if (ntokens != (mxGetM(prhs[7])*mxGetN(prhs[7]))) mexErrMsgTxt("Word and ZIN vectors should have same number of entries");
	}

	// seeding
	seedMT(1 + SEED * 2); // seeding only works on uneven numbers

	/* allocate memory */
	z = ivec(ntokens);

	if (startcond == 1) {
		for (i=0; i<ntokens; i++) z[i] = (int) ZIN[i] - 1;   
	}

	d = ivec(ntokens);
	w = ivec(ntokens);

	// copy over the word and document indices into internal format
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

	phi = ivec(J*W);
	theta = ivec(J*D);

	/* run the learning algorithm */
	FGS(ALPHA, BETA, W, J, D, NN, OUTPUT, ntokens, z, d, w, phi, theta, startcond);

	/* convert the full phi matrix into a sparse matrix */
	nzmaxphi = (int) 0;
	for (i=0; i<W; i++) {
		for (j=0; j<J; j++)
			nzmaxphi += (int) (*(phi + j + i*J)) > 0;
	}  

	/* MAKE THE phi SPARSE MATRIX */
	plhs[0] = mxCreateSparse(J, W, nzmaxphi, mxREAL);
	srphi  = mxGetPr(plhs[0]);
	irphi = mxGetIr(plhs[0]);
	jcphi = mxGetJc(plhs[0]);  
	n = 0;
	for (i=0; i<W; i++) {
		*(jcphi + i) = n;
		for (j=0; j<J; j++) {
			c = (int) *(phi + i*J + j);
			if (c >0) {
				*(srphi + n) = c;
				*(irphi + n) = j;
				n++;
			}
		}    
	}  
	*(jcphi + W) = n;    

	/* MAKE THE theta SPARSE MATRIX */
	nzmaxtheta = (int) 0;
	for (i=0; i<D; i++) {
		for (j=0; j<J; j++)
			nzmaxtheta += (int) (*( theta + j + i*J )) > 0;
	}  

	plhs[1] = mxCreateSparse(J, D, nzmaxtheta, mxREAL );
	srtheta  = mxGetPr(plhs[1]);
	irtheta = mxGetIr(plhs[1]);
	jctheta = mxGetJc(plhs[1]);
	n = 0;
	for (i=0; i<D; i++) {
		*(jctheta + i) = n;
		for (j=0; j<J; j++) {
			c = (int) *(theta + i*J + j);
			if (c >0) {
				*(srtheta + n) = c;
				*(irtheta + n) = j;
				n++;
			}
		}
	}
	*(jctheta + D) = n;

	plhs[2] = mxCreateDoubleMatrix(1, ntokens, mxREAL);
	Z = mxGetPr(plhs[2]);
	for (i=0; i<ntokens; i++) Z[i] = (double) z[i] + 1;
}