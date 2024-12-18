#include <mex.h>
#include <math.h>
#include <matrix.h>

/*
g = mexgradedme(v, u, idxI, idxJ, gtmp, ww, lambda)
*/


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *v, *u, *idxI, *idxJ, *gtmp, *ww, *h, lambda, tmp, tmpu, tmpv, gtmpk, wwk, htmpk;
	mwSize m, n, k;
    mwIndex i, j, kk, l, ik, jk;
	
	v = mxGetPr(prhs[0]);
    u = mxGetPr(prhs[1]);
	idxI = mxGetPr(prhs[2]);
	idxJ = mxGetPr(prhs[3]);
	gtmp = mxGetPr(prhs[4]);
    ww = mxGetPr(prhs[5]);
    lambda = mxGetScalar(prhs[6]);	
	
    k = mxGetM(prhs[0]);
    n = mxGetN(prhs[0]);
    m = mxGetM(prhs[2]);
    
    plhs[0] = mxCreateDoubleMatrix(k, n, mxREAL);
    h = mxGetPr(plhs[0]);
    
    for(j = 0; j < n*k; j++) 
        h[j] = 2*lambda*u[j];
    
    for(kk = 0; kk < m; kk++) 
    {
        i = (mwIndex) idxI[kk];
        j = (mwIndex) idxJ[kk];
        ik = (i-1)*k;
        jk = (j-1)*k;
        gtmpk = gtmp[kk];
        wwk = 8.0 * ww[kk];
        htmpk = 0.0;
        for(l = 0; l < k; l++)
        {
            tmpv = v[ik+l] - v[jk+l];
            tmpu = u[ik+l] - u[jk+l];
            htmpk += wwk * tmpv * tmpu;
        }
        for(l = 0; l < k; l++)
        {
            tmpv = v[ik+l] - v[jk+l];
            tmpu = u[ik+l] - u[jk+l];
            tmp = htmpk * tmpv + gtmpk * tmpu;
            h[ik+l] += tmp;
            h[jk+l] -= tmp;
        }
    }
	
	return;
}
