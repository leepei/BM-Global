#include <mex.h>
#include <math.h>
#include <matrix.h>

/*
g = mexgradedme(xi, idxI, idxJ, gtmp, lambda)
*/


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *v, *idxI, *idxJ, *gtmp, *g, lambda, tmp, gtmpk;
	mwSize m, n, k;
    mwIndex i, j, kk, l, ik, jk;
	
	v = mxGetPr(prhs[0]);
	idxI = mxGetPr(prhs[1]);
	idxJ = mxGetPr(prhs[2]);
	gtmp = mxGetPr(prhs[3]);
    lambda = mxGetScalar(prhs[4]);	
	
    k = mxGetM(prhs[0]);
    n = mxGetN(prhs[0]);
    m = mxGetM(prhs[2]);
    
    plhs[0] = mxCreateDoubleMatrix(k, n, mxREAL);
    g = mxGetPr(plhs[0]);
    
    for(j = 0; j < n; j++) 
    {
        jk = j*k;
        for (i = 0; i < k; i++)
        {
            g[jk+i] = 2*lambda*v[jk+i];
        }
    }
    
    for(kk = 0; kk < m; kk++) 
    {
        i = (mwIndex) idxI[kk];
        j = (mwIndex) idxJ[kk];
        ik = (i-1)*k;
        jk = (j-1)*k;
        gtmpk = gtmp[kk];
        for(l = 0; l < k; l++)
        {
            tmp = gtmpk*(v[ik+l] - v[jk+l]);
            g[ik+l] += tmp;
            g[jk+l] -= tmp;
        }
    }
	
	return;
}
