#include <mex.h>
#include <math.h>
#include <matrix.h>

/*
g = mexgradedme(xi, idxI, idxJ, gtmp, lambda)
*/


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *xi, *idxI, *idxJ, *gtmp, *g, lambda, tmp;
	mwSize m, n;
    mwIndex i, j, k;
	
	xi = mxGetPr(prhs[0]);
	idxI = mxGetPr(prhs[1]);
	idxJ = mxGetPr(prhs[2]);
	gtmp = mxGetPr(prhs[3]);
    lambda = mxGetScalar(prhs[4]);
	
	m = mxGetM(prhs[2]);
    n = mxGetM(prhs[0]);
    
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    g = mxGetPr(plhs[0]);
    
    for(k = 0; k < n; k++)
    {
        g[k] = lambda*xi[k];
    }
    
    for(k = 0; k < m; k++) 
    {
        i = (mwIndex) idxI[k]-1;
        j = (mwIndex) idxJ[k]-1;
        tmp = 2.0 * gtmp[k] * (xi[i] - xi[j]);
        g[i] += tmp;
        g[j] -= tmp;
    }
	
	return;
}
