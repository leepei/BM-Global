/* This for computing the loss term of \| P_{Omega} (W H^T - X)\|_F^2, where Omega is the observed entries
 * The key idea for saving time is to avoid even any memory copying in forming W, H, and X by using the pointers from MATLAB directly
 * The computation then goes through all entries in Omega to do the sum in parallel through openmp
 * The dense matrices W, H are column major, but the access here is row major so parallelization might not help much as the memory access
 * pattern for sure incurs a lot of cache misses and the bottleneck is memory instead of computation
 * But doing transpose first is meaningless because we are just shifting the nonsequential memory access to a different place
 */

#include "mex.h"
#include <omp.h>
#include <cstring>
struct mat_t_const
{
	size_t m,n;
	double *entries;
};

class smat_t_const{
	public:
		size_t rows, cols;
		size_t nnz;
		double *val, *val_t;
		double *weight, *weight_t;
		size_t *col_ptr, *row_ptr;
		size_t *col_nnz, *row_nnz;
		size_t *row_idx, *col_idx;
		bool mem_alloc_by_me, with_weights;
		smat_t_const():mem_alloc_by_me(false), with_weights(false){ }
		smat_t_const(const smat_t_const & m){ *this = m; mem_alloc_by_me = false;}

		smat_t_const(size_t m, size_t n, size_t *ir, size_t *jc, double *v, size_t *ir_t, size_t *jc_t, double *v_t):
			rows(m), cols(n), mem_alloc_by_me(false),
			row_idx(ir), col_ptr(jc), col_idx(ir_t), row_ptr(jc_t), val(v), val_t(v_t){
			if(col_ptr[n] != row_ptr[m])
				fprintf(stderr,"Error occurs! two nnz do not match (%ld, %ld)\n", col_ptr[n], row_ptr[n]);
			nnz = col_ptr[n];
		}

		void from_mpi(){
			mem_alloc_by_me=true;
		}
		~smat_t_const(){
			if(mem_alloc_by_me) {
				//puts("Warnning: Somebody just free me.");
				free(val); free(val_t);
				free(row_ptr);free(row_idx);
				free(col_ptr);free(col_idx);
				if(with_weights) { free(weight); free(weight_t);}
			}
		}
};

double comp_and_sum_residual(smat_t_const &R, mat_t_const &A, mat_t_const &B, )
{
	int K = A.n;
	int M = R.rows;
	int N = R.cols;
	double  loss = 0;
#pragma omp parallel for reduction(+:loss) schedule(dynamic,256)
	for(int i=0; i<M; i++)
	{
		for(int idx=R.row_ptr[i]; idx<R.row_ptr[i+1]; idx++)
		{
			int j = R.col_idx[idx];
			double rij = R.val_t[idx];
			for(int kk=0; kk<K; kk++)
				rij -= A.entries[kk*M + i]*B.entries[kk*N + j];
			loss += rij*rij;
		}
	}
	return loss;
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mat_t_const W, H;
	smat_t_const R((size_t) mxGetM(prhs[0]), (size_t) mxGetN(prhs[0]),(size_t *) mxGetIr(prhs[0]), (size_t *) mxGetJc(prhs[0]),mxGetPr(prhs[0]), (size_t *) mxGetIr(prhs[1]), (size_t *) mxGetJc(prhs[1]),mxGetPr(prhs[1]));
	const mxArray *mxW = prhs[2];
	size_t rows = mxGetM(mxW), cols = mxGetN(mxW);
	double *val = mxGetPr(mxW);
	int threads;
	int num_threads_old = omp_get_num_threads();
	if (nrhs == 5)
	{
		threads = int(mxGetScalar(prhs[4]));
		omp_set_num_threads(threads);
	}
	W.m = rows;
	W.n = cols;
	W.entries = val;
	const mxArray *mxH = prhs[3];
	rows = mxGetM(mxH);
	cols = mxGetN(mxH);
	val = mxGetPr(mxH);
	H.m = rows;
	H.n = cols;
	H.entries = val;
	double loss = comp_and_sum_residual(R, W, H);
	plhs[0] = mxCreateDoubleScalar(loss);
	return;
}
