#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mex.h>
#include "topiclib.cpp"

// Syntax
//   [ PHI , THETA , Z , X ] = ATMGS( WD, AD , J , N , ALPHA , BETA , SEED , OUTPUT )

// Syntax
//   [ PHI , THETA , Z , X ] = ATMGS( WD, AD , J , N , ALPHA , BETA , SEED , OUTPUT , ZIN , XIN )

int NAMAX = 100; // maximum number of authors on a document  

void ATMGS( double ALPHA, double BETA, int W, int J, int D, int NN, int OUTPUT, int n, int *z, int *d, int *w, 
	int *x, int *phi, int *theta, int *phitot, int *atot, int *order, 
	double *probs, mwIndex *irad, mwIndex *jcad, int startcond)
{
	int wi, di, i, ii, j, k, topic, rp, temp, iter, wioffset, aioffset, i_start, i_end, i_pick, author;
	int nauthors, kk;
	double totprob, WBETA, KALPHA, r, max;

	if (startcond==1) {
		/* start from previous state */
		for (i=0; i<n; i++) {
			wi     = w[i];
			di     = d[i];
			topic  = z[i];

			// i_start and i_end are the start and end indices of the sparse AD matrix for document di
			i_start = jcad[di];
			i_end   = jcad[di+1];
			nauthors = i_end - i_start;

			if (nauthors==1)
				author = irad[i_start];
			else  {
				// pick a random number between i_start and i_end-1
				i_pick = i_start + (int) (nauthors*drand());

				// which author does this correspond to?
				author = irad[i_pick];
			}

			author = x[i];

			// update counts for this author
			theta[author*J + topic]++; // increment theta count matrix
			atot[author]++;		
		}
	}

	if (startcond==0) {
		/* random initialization */
		for (i=0; i<n; i++)
		{
			wi = w[i];
			di = d[i];
			// pick a random topic 0..J-1
			topic = (int) (J*drand());
			z[i] = topic; // assign this word token to this topic

			// i_start and i_end are the start and end indices of the sparse AD matrix for document di
			i_start = jcad[di];
			i_end   = jcad[di+1];
			nauthors = i_end - i_start;

			if (nauthors==1)
				author = irad[i_start];
			else  {
				// pick a random number between i_start and i_end-1
				i_pick = i_start + (int) (nauthors*drand());

				// which author does this correspond to?
				author = irad[i_pick];
			}

			x[i] = author;

			// update counts for this author
			theta[author*J + topic]++; // increment theta count matrix
			atot[author]++;		
		}
	}

	for (i=0; i<n; i++) order[i]=i; // fill with increasing series
	for (i=0; i<(n-1); i++) {
		// pick a random integer between i and nw
		rp = i + (int) ((n-i)*drand());

		// switch contents on position i and position rp
		temp = order[rp];
		order[rp]=order[i];
		order[i]=temp;
	}

	WBETA = W*BETA;
	KALPHA = J*ALPHA;

	for (iter=0; iter<NN; iter++) {
		if (OUTPUT >=1) {
			if ((iter % 10)==0) mexPrintf( "\tIteration %d of %d\n" , iter , NN );
			if ((iter % 10)==0) mexEvalString("drawnow;");
		}
		for (ii=0; ii<n; ii++) {
			i = order[ii]; // current word token to assess

			wi     = w[i]; // current word index
			di     = d[i]; // current document index  
			topic  = z[i]; // current topic assignment for word token
			author = x[i]; // current author assignment for word token

			i_start  = jcad[di];
			i_end    = jcad[di+1];
			nauthors = i_end - i_start;

			atot[author]--;           
			wioffset = wi*J;
			aioffset = author*J;
			theta[aioffset+topic]--;

			totprob = (double) 0;
			kk = 0;
			for (k=0; k<nauthors; k++) {
				author = irad[i_start + k]; // current author index under consideration
				aioffset = author*J;		    

				for (j=0; j<J; j++) 
				{	  
					// probs[j][k] contains the (unnormalized) probability of assigning this word token to topic j and author k
					probs[kk] = 
						((double) phi[wioffset+j] + (double) BETA)/((double) phitot[j] + (double) WBETA) *	             
						((double) theta[aioffset+j] + (double) ALPHA)/((double) atot[author] + (double) KALPHA );

					totprob += probs[kk];
					kk++;
				}	  
			}

			// sample a topic from the distribution
			r = totprob*drand();
			max = probs[0];
			if (nauthors==1) {
				topic = 0;
				while (r>max) {
					topic++;
					max += probs[topic];
				}
			} else {
				kk = 0;
				k  = 0;
				topic = 0;
				while (r>max) {
					kk++;
					topic++;
					if (topic == J) {
						k++;
						topic = 0;
					}
					max += probs[kk];
				}
				author = irad[i_start + k]; // sampled author
			}

			z[i] = topic; // assign current word token i to topic j
			x[i] = author;

			aioffset = author*J;
			atot[author]++;
			theta[aioffset+topic]++;
		}
	}

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *srphi, *srtheta, *srad, *sr, *probs, *WS, *DS, *ZIN, *XIN, *Z, *X;
	double ALPHA, BETA;
	mwIndex *irphi, *jcphi, *irtheta, *jctheta, *irad, *jcad, *ir, *jc;
	int *z, *d, *w, *x, *order, *phi, *theta, *phitot, *atot;
	int W, J, D, A, NN, SEED, OUTPUT, nzmax, nzmaxad, nzmaxphi, nzmaxtheta, ntokens;
	int i, j, k, c, n, nt, wi, ci, ntcount, di;
	int i_start, i_end, a, nauthors, startcond;

	// Syntax
	//   [ PHI , THETA , Z , X ] = ATMGS( WD, AD , J , N , ALPHA , BETA , SEED , OUTPUT )

	// Syntax
	//   [ PHI , THETA , Z , X ] = ATMGS( WD, AD , J , N , ALPHA , BETA , SEED , OUTPUT , ZIN , XIN )


	/* Check for proper number of arguments. */
	if (nrhs < 8) {
		mexErrMsgTxt("At least 8 input arguments required");
	} else if (nlhs < 1) {
		mexErrMsgTxt("At least 1 output arguments required");
	}

	startcond = 0;
	if (nrhs > 8) startcond = 1;

	/* dealing with sparse array WD */
	if (mxIsDouble(prhs[0]) != 1) mexErrMsgTxt("WD must be a double precision matrix");
	sr = mxGetPr(prhs[0]);
	ir = mxGetIr(prhs[0]);
	jc = mxGetJc(prhs[0]);
	nzmax = (int) mxGetNzmax(prhs[0]);
	W = (int) mxGetM(prhs[0]);
	D = (int) mxGetN(prhs[0]);

	// get the number of tokens
	ntokens = (int) 0;
	for  (i=0; i<nzmax; i++) ntokens += (int) sr[i];
	if (ntokens == 0) mexErrMsgTxt("word vector is empty"); 

	d = ivec(ntokens);
	w = ivec(ntokens);

	// copy over the word and document indices into internal format
	k = (int) 0;
	for (di=0; di<D; di++) {
		for (i=jc[di]; i<jc[di+1]; i++) {
			wi = (int) ir[i];
			ci = (int) sr[i];
			for (j=0; j<ci; j++) {
				d[k] = di;
				w[k] = wi;
				k++;
			}
		}
	}

	if (ntokens != k) mexErrMsgTxt("Fail to read data");

	n = ntokens;

	if ((mxIsSparse( prhs[ 1 ] ) != 1) || (mxIsDouble( prhs[ 1 ] ) != 1))
		mexErrMsgTxt("Input matrix must be a sparse double precision matrix");

	/* dealing with sparse array AD */
	srad = mxGetPr(prhs[1]);
	irad = mxGetIr(prhs[1]);
	jcad = mxGetJc(prhs[1]);
	nzmaxad = mxGetNzmax(prhs[1]);
	A = mxGetM( prhs[1] );
	if (mxGetN( prhs[1] ) != D) mexErrMsgTxt("The number of columns in WD and AD matrix must be equal" );

	// get phi
	srphi = mxGetPr(prhs[2]);
	irphi = mxGetIr(prhs[2]);
	jcphi = mxGetJc(prhs[2]);
	nzmaxphi = (int) mxGetNzmax(prhs[2]);
	if (W != (int) mxGetN(prhs[2])) mexErrMsgTxt("Vocabulary size mismatches"); ;
	J = (int) mxGetM(prhs[2]);
	phi  = (int *) mxCalloc(J*W , sizeof(int));
	phitot = (int *) mxCalloc(J, sizeof(int));
	for (wi=0; wi<W; wi++) {
		for (i=jcphi[wi]; i<jcphi[wi+1]; i++) {
			j = (int) irphi[i];
			phi[wi*J + j] = (int) srphi[i];
			phitot[j] += (int) srphi[i];
		}
	}

	NN    = (int) mxGetScalar(prhs[3]);
	if (NN<0) mexErrMsgTxt("Number of iterations must be greater than zero");

	ALPHA = (double) mxGetScalar(prhs[4]);
	if (ALPHA<=0) mexErrMsgTxt("ALPHA must be greater than zero");

	BETA = (double) mxGetScalar(prhs[5]);
	if (BETA<=0) mexErrMsgTxt("BETA must be greater than zero");

	SEED = (int) mxGetScalar(prhs[6]);
	// set the seed of the random number generator

	OUTPUT = (int) mxGetScalar(prhs[7]);

	if (startcond == 1) {
		ZIN = mxGetPr(prhs[8]);
		if (ntokens != (mxGetM(prhs[8])*mxGetN(prhs[8]))) mexErrMsgTxt("WS and ZIN vectors should have same number of entries");

		XIN = mxGetPr(prhs[9]);
		if (ntokens != (mxGetM(prhs[9])*mxGetN(prhs[9]))) mexErrMsgTxt("WS and XIN vectors should have same number of entries");
	}

	// seeding
	seedMT(1 + SEED*2); // seeding only works on uneven numbers

	// check entries of AD matrix
	for (i=0; i<nzmaxad; i++) {
		nt = (int) srad[i];    
		if ((nt<0) || (nt>1)) mexErrMsgTxt("Entries in AD matrix can only be 0 or 1");
	}

	/* allocate memory */
	z = ivec(ntokens);
	x = ivec( ntokens);

	if (startcond == 1) {
		for (i=0; i<ntokens; i++) {
			z[i] = (int) ZIN[i] - 1; 
			x[i] = (int) XIN[i] - 1; 
		}
	}

	order = ivec(ntokens);
	theta = ivec(J*A);
	phitot = ivec(J);
	atot = ivec(A);
	probs = dvec(J*NAMAX);

	/* check that every document has some authors */
	for (j=0; j<D; j++) {
		i_start = jcad[j];
		i_end   = jcad[j+1];  
		nauthors = i_end - i_start;
		if (nauthors == 0) mexErrMsgTxt("There are some documents without authors in AD matrix ");
		if (nauthors > NAMAX) mexErrMsgTxt("Too many authors in some documents ... reached the NAMAX limit");
	}

	/* run the model */
	ATMGS(ALPHA, BETA, W, J, D, NN, OUTPUT, n, z, d, w, x, phi, theta, phitot, atot, order, probs, irad, jcad, startcond);

	// create sparse matrix theta
	nzmaxtheta = 0;
	for (i=0; i<A; i++) {
		for (j=0; j<J; j++)
			nzmaxtheta += (int) ( *( theta + j + i*J )) > 0;
	}

	plhs[0] = mxCreateSparse(J, A, nzmaxtheta, mxREAL);
	srtheta  = mxGetPr(plhs[0]);
	irtheta = mxGetIr(plhs[0]);
	jctheta = mxGetJc(plhs[0]);

	n = 0;
	for (i=0; i<A; i++) {
		*( jctheta + i ) = n;
		for (j=0; j<J; j++) {
			c = (int) *( theta + i*J + j );
			if (c >0) {
				*( srtheta + n ) = c;
				*( irtheta + n ) = j;
				n++;
			}
		}
	}

	*(jctheta + A) = n;

	plhs[1] = mxCreateDoubleMatrix(1, ntokens, mxREAL);
	Z = mxGetPr(plhs[1]);
	for (i=0; i<ntokens; i++) Z[i] = (double) z[i] + 1;

	plhs[2] = mxCreateDoubleMatrix(1, ntokens, mxREAL);
	X = mxGetPr(plhs[2]);
	for (i=0; i<ntokens; i++) X[i] = (double) x[i] + 1;
}
