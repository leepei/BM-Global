#include <mex.h>
#include <math.h>
#include <matrix.h>

/*
y = mexlinmapedme(vt, idxI, idxJ)
y = sum_{i,j}<Eij, vv'>
*/


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *vt, *y, *idxI, *idxJ;
	mwSize m, k;
	
	vt = mxGetPr(prhs[0]);
	idxI = mxGetPr(prhs[1]);
	idxJ = mxGetPr(prhs[2]);
	
	k = mxGetM(prhs[0]);
	m = mxGetM(prhs[1]);
	
	plhs[0] = mxCreateDoubleMatrix(m, 1, mxREAL);
	y = mxGetPr(plhs[0]);
	
#pragma omp parallel for schedule(dynamic,16)
	for (mwIndex kk = 0; kk < m; kk++) 
	{
		mwIndex i = (mwIndex) idxI[kk];
		mwIndex j = (mwIndex) idxJ[kk];
        mwIndex ik = (i-1)*k;
        mwIndex jk = (j-1)*k;
		double vij = 0.0;
		for (mwIndex l = 0; l < k; l++)
		{
			double tmp = vt[ik+l] - vt[jk+l];
			vij += (tmp * tmp);
		}
		y[kk] = vij;
	}
	
	return;
}
