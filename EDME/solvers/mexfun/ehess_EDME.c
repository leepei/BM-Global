#include <mex.h>
#include <math.h>
#include <matrix.h>
#include <string.h>
#include <omp.h>

/*
g = mexgradedme(v, u, idxI, idxJ, gtmp, ww, lambda)
*/


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	int threads;
	int num_threads_old = omp_get_num_threads();
	if (nrhs == 8)
	{
		threads = int(mxGetScalar(prhs[7]));
		omp_set_num_threads(threads);
	}
	double *v, *u, *idxI, *idxJ, *gtmp, *ww, *h;
	mwSize m, n, k;
	
	v = mxGetPr(prhs[0]);
	u = mxGetPr(prhs[1]);
	idxI = mxGetPr(prhs[2]);
	idxJ = mxGetPr(prhs[3]);
	gtmp = mxGetPr(prhs[4]);
	ww = mxGetPr(prhs[5]);
	h = mxGetPr(prhs[6]);	
	
	k = mxGetM(prhs[0]);
	n = mxGetN(prhs[0]);
	m = mxGetM(prhs[2]);
	double **tmph = new double*[threads];
	double **tmpvecv = new double*[threads];
	double **tmpvecu = new double*[threads];

#pragma omp parallel for schedule(static,1)
	for (int i=0;i<threads;i++)
	{
		int thread_id = omp_get_thread_num();
		tmph[thread_id] = new double[k*n];
		tmpvecv[thread_id] = new double[k];
		tmpvecu[thread_id] = new double[k];
		memset(tmph[thread_id], 0, sizeof(double) * k * n);
	}

#pragma omp parallel for schedule(guided,16)
	for(mwIndex kk = 0; kk < m; kk++)
	{
		int thread_id = omp_get_thread_num();
		mwIndex i = (mwIndex) idxI[kk];
		mwIndex j = (mwIndex) idxJ[kk];
		mwIndex ik = (i-1)*k;
		mwIndex jk = (j-1)*k;
		double gtmpk = gtmp[kk];
		double wwk = 8*ww[kk];
		double htmpk = 0.0;
		for(mwIndex l = 0; l < k; l++)
		{
			tmpvecv[thread_id][l] = v[ik+l] - v[jk+l];
			tmpvecu[thread_id][l] = u[ik+l] - u[jk+l];
			htmpk += tmpvecv[thread_id][l] * tmpvecu[thread_id][l];
		}
		htmpk *= wwk;
		for(mwIndex l = 0; l < k; l++)
		{
			double tmp = htmpk * tmpvecv[thread_id][l] + gtmpk * tmpvecu[thread_id][l];
			tmph[thread_id][ik+l] += tmp;
			tmph[thread_id][jk+l] -= tmp;
		}
	}
#pragma omp parallel for schedule(guided,16)
	for (mwIndex i=0;i<k*n;i++)
		for (int id=0;id<threads;id++)
			h[i] += tmph[id][i];

	for (int i=0;i<threads;i++)
	{
		delete[] tmph[i];
		delete[] tmpvecv[i];
		delete[] tmpvecu[i];
	}

	delete[] tmph;
	delete[] tmpvecv;
	delete[] tmpvecu;
	
	return;
}
