#include "mex.h"
#include "stdio.h" 
#include "string.h"
#include "stdlib.h"
#define MAX 50000 // maximum length of the line is 50000 characters

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	mwIndex *ir, *jc;
	double *pr;
	char *filename, *pch, mystring[MAX];
	int buflen, d, i, D, W, NNZ;
	FILE *doc;

	/* Check for proper number of arguments. */
	if (nrhs != 1) {
		mexErrMsgTxt("Only one input argument required");
	} else if (nlhs != 1) {
		mexErrMsgTxt("Only one output argument required");
	}

	// open document file
	buflen = (int) (1 + (mxGetM(prhs[0])*mxGetN(prhs[0])));
	filename = (char *) mxCalloc(buflen, sizeof(char));
	mxGetString( prhs[0], filename, buflen);
	doc = fopen(filename, "r"); 
	if (doc == NULL) mexErrMsgTxt("Text file cannot be found.");

	// read the file header
	fgets(mystring , sizeof(mystring) , doc);
	if ((mystring != NULL) && (atoi(mystring) != 0)) {
		D = atoi(mystring);
		mexPrintf( "#Document: %d \n", D);
	} else {
		mexErrMsgTxt("File header error.");
	}
	
	fgets(mystring , sizeof(mystring) , doc);
	if ((mystring != NULL) && (atoi(mystring) != 0)){
		W = atoi(mystring);
		mexPrintf( "#Vocabulary: %d \n", W);
	} else {
		mexErrMsgTxt("File header error.");
	}
	
	fgets(mystring , sizeof(mystring) , doc);
	if ((mystring != NULL) && (atoi(mystring) != 0)){
		NNZ = atoi(mystring);
		mexPrintf( "#NNZ: %d \n", NNZ);
	} else {
		mexErrMsgTxt("File header error.");
	}

	// create W x D sparse matrix for data
	plhs[0] = mxCreateSparse(W, D, NNZ, mxREAL);
	pr  = (double *) mxGetPr(plhs[0]);
	ir = mxGetIr(plhs[0]);
	jc = mxGetJc(plhs[0]);

	// copy data to sparse matrix
	i = 0;
	jc[0] = 0;
	while (fgets(mystring , sizeof(mystring) , doc) != NULL) {	
		pch = strtok(mystring, " ");
		if ((pch != NULL) && (atoi(pch) != 0)) {
			d = atoi(pch);
			pch = strtok(NULL, " ");
			ir[i] = atoi(pch) - 1;
			pch = strtok(NULL, " ");
			pr[i] = atoi(pch);
			i++;
			jc[d] = i;
		}
	}
	fclose(doc);
}
