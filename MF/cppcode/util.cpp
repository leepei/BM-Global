#include "util.h"
#include <errno.h>
#include <string.h>
#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))

#ifdef __cplusplus
extern "C" {
#endif
extern double ddot_(int *, double *, int *, double *, int *);
#ifdef __cplusplus
}
#endif


// load utility for CCS RCS
void load(const char* srcdir, smat_t &R, testset_t &T, bool with_weights){
	// add testing later
	char filename[1024], buf[1024];
	sprintf(filename,"%s/meta",srcdir);
	FILE *fp = fopen(filename,"r");
	size_t m, n, nnz;
	fscanf(fp, "%ld %ld", &m, &n);

	fscanf(fp, "%ld %s", &nnz, buf);
	sprintf(filename,"%s/%s", srcdir, buf);
	R.load(m, n, nnz, filename, with_weights);

	if(fscanf(fp, "%ld %s", &nnz, buf)!= EOF){
		sprintf(filename,"%s/%s", srcdir, buf);
		T.load(m, n, nnz, filename);
	}
	fclose(fp);
	//double bias = R.get_global_mean(); R.remove_bias(bias); T.remove_bias(bias);
	return ;
}

// Save a mat_t A to a file in row_major order.
// row_major = true: A is stored in row_major order,
// row_major = false: A is stored in col_major order.
void save_mat_t(mat_t A, FILE *fp, bool row_major)
{
	if (fp == NULL) 
		fprintf(stderr, "output stream is not valid.\n");
	size_t m = A.m;
	size_t n = A.n;
	fwrite(&m, sizeof(size_t), 1, fp);
	fwrite(&n, sizeof(size_t), 1, fp);
	if (row_major)
		fwrite(A.entries, sizeof(double), m*n, fp);
	else
	{
		vec_t buf(m*n);
		size_t idx = 0;
		for(size_t i = 0; i < m; ++i) 
			for(size_t j = 0; j < n; ++j)
				buf[idx++] = A.entries[j*m + i];
		fwrite(&buf[0], sizeof(double), m*n, fp);
	}
}

// Load a matrix from a file rstored in row major) and return a mat_t matrix 
// row_major = true: the returned A is stored in row_major order,
// row_major = false: the returened A  is stored in col_major order.
mat_t load_mat_t(FILE *fp, bool row_major){
	if (fp == NULL) 
		fprintf(stderr, "input stream is not valid.\n");
	size_t m, n; 
	fread(&m, sizeof(size_t), 1, fp);
	fread(&n, sizeof(size_t), 1, fp);
	mat_t A;
	A.m = m;
	A.n = n;
	A.entries = new double[m*n];
	if (row_major)
		fread(A.entries, sizeof(double), m*n, fp);
	else
	{
		vec_t buf(m*n);
		fread(&buf[0], sizeof(double), m*n, fp);
		size_t idx = 0;
		for(size_t i = 0; i < m; ++i)
			for(size_t j = 0; j < n; ++j)
				A.entries[j*m+i] = buf[idx++];
	}
	return A;
}

//W H^T = X, row major, so W is m*k and H is n*k
//But in implemention H^T is k*n and accessed in the storage pattern
//Every time we fix k and update a whole row
//W[0...m-1]: 1st row
//W[m...2m-1]: 2nd
//W[(k-1)*m...k*m-1] last row
void initial_col(mat_t &X, size_t k, size_t n){
	X.m = n;
	X.n = k;
	X.entries = new double[k*n];
	srand48(0L);
	for(size_t i = 0; i < k*n; ++i)
			X.entries[i] = 0.1*drand48();
}

double dot(vec_t &a, vec_t &b){
	double ret = 0;
	int n = a.size();
	int inc = 1;
	for (int i=0;i<n;i++)
		ret += a[i] * b[i];
	return ret;
}

double dot(mat_t &W, int i, mat_t &H, int j){ //(W^T H)_{i,j}
	int k = W.n;
	int m1 = W.m;
	int m2 = H.m;
	double ret = 0;
//	double ret = ddot_(&k, W.entries + i, &m1, H.entries + j, &m2);
	for(int t = 0; t < k; ++t)
		ret += W.entries[m1 * t + i] * H.entries[m2 * t + j];
	return ret;
}

double dot(mat_t &W, int i, vec_t &H_j){
	int k = H_j.size();
	int m = W.m;
	int inc = 1;
	double ret = 0;
//	double ret = ddot_(&k, W.entries + i, &m, H_j.data(), &inc);
	for(int t = 0; t < k; ++t)
		ret+=W.entries[t*m + i]*H_j[t];
	return ret;
}

inline double norm(vec_t &a){
	return dot(a,a);
}
double norm(mat_t &M)
{
	int size = M.m * M.n;
	int inc = 1;
//	return ddot_(&size, M.entries, &inc, M.entries, &inc);
	double ret = 0;
	for (int i=0;i<size;i++)
		ret += M.entries[i] * M.entries[i];
	return ret;
}

double calobj(const smat_t &R, mat_t &W, mat_t &H, double lambda){
	double loss = 0;
	int k = W.n;
	for(size_t c = 0; c < R.cols; ++c){
		for(size_t idx = R.col_ptr[c]; idx < R.col_ptr[c+1]; ++idx){
			double diff = - R.val[idx];
			diff += dot(W,R.row_idx[idx], H,c);
			loss += (R.with_weights? R.weight[idx] : 1.0) * diff*diff;
		}
	}
	double reg = lambda*(norm(W) + norm(H));
	return loss + reg;
}

double calrmse_r1(testset_t &testset, double *Wt, double *Ht, double *oldWt, double *oldHt){//outer product of the updated terms
	size_t nnz = testset.nnz;
	double rmse = 0, err;
#pragma omp parallel for reduction(+:rmse)
	for(size_t idx = 0; idx < nnz; ++idx){
		testset[idx].v -= Wt[testset[idx].i]*Ht[testset[idx].j] - oldWt[testset[idx].i]*oldHt[testset[idx].j];
		rmse += testset[idx].v*testset[idx].v;
	}
	return sqrt(rmse/nnz);
}

double init_testset_residual(testset_t &testset, mat_t &W, mat_t &H){
	size_t nnz = testset.nnz;
	double rmse = 0, err;
	for(size_t idx = 0; idx < nnz; ++idx){
		testset[idx].v -= dot(W, testset[idx].i, H, testset[idx].j);
		err = testset[idx].v;
		rmse += err*err;
	}
	return sqrt(rmse/nnz);
}
