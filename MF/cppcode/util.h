#ifndef MATUTIL
#define MATUTIL
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <utility>
#include <map>
#include <queue>
#include <set>
#include <vector>
#include <cmath>
#include <omp.h>
#include <assert.h>

#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))

enum {ROWMAJOR, COLMAJOR};

using namespace std;
class rate_t;
class rateset_t; 
class RateComp;
class smat_t;
class testset_t;
typedef vector<double> vec_t;
struct mat_t
{
	size_t m,n;
	double *entries;
};
//typedef vector<vec_t> mat_t;
void load(const char* srcdir, smat_t &R, testset_t &T, bool with_weights = false);
void save_mat_t(mat_t A, FILE *fp, bool row_major = true);
mat_t load_mat_t(FILE *fp, bool row_major = true);
void initial_col(mat_t &X, size_t k, size_t n);
void reinitial(mat_t &X, size_t k, size_t n);
double dot(vec_t &a, vec_t &b);
double dot(mat_t &W, int i, mat_t &H, int j);
double dot(mat_t &W, int i, vec_t &H_j);
double norm(vec_t &a);
double norm(mat_t &M);
double calobj(const smat_t &R, mat_t &W, mat_t &H, double lambda);
double calrmse_r1(testset_t &testset, double *Wt, double *Ht, double *oldWt, double *oldHt);

class rate_t{
	public:
		int i, j; double v, weight;
		rate_t(int ii=0, int jj=0, double vv=0, double ww=1.0): i(ii), j(jj), v(vv), weight(ww){}
};

class entry_iterator_t{
	private:
		FILE *fp;
		char buf[1000];
	public:
		bool with_weights;
		size_t nnz;
		entry_iterator_t():nnz(0),fp(NULL), with_weights(false){}
		entry_iterator_t(size_t nnz_, const char* filename, bool with_weights_=false) {
			nnz = nnz_;
			fp = fopen(filename,"r");
			with_weights = with_weights_;
		}
		size_t size() {return nnz;}
		virtual rate_t next() {
			int i = 1, j = 1;
			double v = 0, w = 1.0;
			if (nnz > 0) {
				fgets(buf, 1000, fp);
				if (with_weights)
					sscanf(buf, "%d %d %lf %lf", &i, &j, &v, &w);
				else 
					sscanf(buf, "%d %d %lf", &i, &j, &v);
				--nnz;
			} else {
				fprintf(stderr,"Error: no more entry to iterate !!\n");
			}
			return rate_t(i-1,j-1,v,w);
		}
		virtual ~entry_iterator_t(){
			if (fp) fclose(fp);
		}
};



// Comparator for sorting rates into row/column comopression storage
class SparseComp {
	public:
		const size_t *row_idx;
		const size_t *col_idx;
		SparseComp(const size_t *row_idx_, const size_t *col_idx_, bool isRCS_=true) {
			row_idx = (isRCS_)? row_idx_: col_idx_;
			col_idx = (isRCS_)? col_idx_: row_idx_;
		}
		bool operator()(size_t x, size_t y) const {
			return  (row_idx[x] < row_idx[y]) || ((row_idx[x] == row_idx[y]) && (col_idx[x]<= col_idx[y]));
		}
};

// Sparse matrix format CCS & RCS
// Access column fomat only when you use it..
class smat_t{
	public:
		size_t rows, cols;
		size_t nnz;
		double *val, *val_t;
		double *weight, *weight_t;
		size_t *col_ptr, *row_ptr;
		size_t *col_nnz, *row_nnz;
		size_t *row_idx, *col_idx;    // condensed
		//size_t *row_idx, *col_idx; // for matlab
		bool mem_alloc_by_me, with_weights;
		smat_t():mem_alloc_by_me(false), with_weights(false){ }
		smat_t(const smat_t& m){ *this = m; mem_alloc_by_me = false;}

		// For matlab (Almost deprecated)
		smat_t(size_t m, size_t n, size_t *ir, size_t *jc, double *v, size_t *ir_t, size_t *jc_t, double *v_t):
		//smat_t(size_t m, size_t n, size_t *ir, size_t *jc, double *v, size_t *ir_t, size_t *jc_t, double *v_t):
			rows(m), cols(n), mem_alloc_by_me(false), 
			row_idx(ir), col_ptr(jc), col_idx(ir_t), row_ptr(jc_t) {
			if(col_ptr[n] != row_ptr[m]) 
				fprintf(stderr,"Error occurs! two nnz do not match (%ld, %ld)\n", col_ptr[n], row_ptr[n]);
			nnz = col_ptr[n];
			val = MALLOC(double, nnz); val_t = MALLOC(double, nnz);
			memcpy(val, v, sizeof(val) * nnz);
			memcpy(val_t, v_t, sizeof(val_t) * nnz);
		}

		void from_mpi(){
			mem_alloc_by_me=true;
		}
		void print_mat(int host){
			for(int c = 0; c < cols; ++c) if(col_ptr[c+1]>col_ptr[c]){
				printf("%d: %ld at host %d\n", c, col_ptr[c+1]-col_ptr[c],host);
			}
		}
		void load(size_t _rows, size_t _cols, size_t _nnz, const char* filename, bool with_weights = false){
			entry_iterator_t entry_it(_nnz, filename, with_weights);
			load_from_iterator(_rows, _cols, _nnz, &entry_it);
		}
		void load_from_iterator(size_t _rows, size_t _cols, size_t _nnz, entry_iterator_t* entry_it) {
			rows =_rows,cols=_cols,nnz=_nnz;
			mem_alloc_by_me = true;
			with_weights = entry_it->with_weights;
			val = MALLOC(double, nnz); val_t = MALLOC(double, nnz);
			if(with_weights) { weight = MALLOC(double, nnz); weight_t = MALLOC(double, nnz); }
			row_idx = MALLOC(size_t, nnz); col_idx = MALLOC(size_t, nnz);  // switch to this for memory
			row_ptr = MALLOC(size_t, rows+1); col_ptr = MALLOC(size_t, cols+1);
			memset(row_ptr,0,sizeof(size_t)*(rows+1));
			memset(col_ptr,0,sizeof(size_t)*(cols+1));

			/*
			 * Assume ratings are stored in the row-majored ordering
			for(size_t idx = 0; idx < _nnz; idx++){
				rate_t rate = entry_it->next();
				row_ptr[rate.i+1]++;
				col_ptr[rate.j+1]++;
				col_idx[idx] = rate.j;
				val_t[idx] = rate.v;
			}*/

			// a trick here to utilize the space the have been allocated 
			vector<size_t> perm(_nnz);
			size_t *tmp_row_idx = col_idx;
			size_t *tmp_col_idx = row_idx;
			double *tmp_val = val;
			double *tmp_weight = weight;
			for(size_t idx = 0; idx < _nnz; idx++){
				rate_t rate = entry_it->next();
				row_ptr[rate.i+1]++;
				col_ptr[rate.j+1]++;
				tmp_row_idx[idx] = rate.i; 
				tmp_col_idx[idx] = rate.j;
				tmp_val[idx] = rate.v;
				if(with_weights) 
					tmp_weight[idx] = rate.weight;
				perm[idx] = idx;
			}
			// sort entries into row-majored ordering
			sort(perm.begin(), perm.end(), SparseComp(tmp_row_idx, tmp_col_idx, true));
			// Generate CRS format
			for(size_t idx = 0; idx < _nnz; idx++) {
				val_t[idx] = tmp_val[perm[idx]];
				col_idx[idx] = tmp_col_idx[perm[idx]];
				if(with_weights)
					weight_t[idx] = tmp_weight[idx];
			}

			// Calculate nnz for each row and col
			for(size_t r=1; r<=rows; ++r) {
				row_ptr[r] += row_ptr[r-1];
			}
			for(size_t c=1; c<=cols; ++c) {
				col_ptr[c] += col_ptr[c-1];
			}
			// Transpose CRS into CCS matrix
			for(size_t r=0; r<rows; ++r){
				for(size_t i = row_ptr[r]; i < row_ptr[r+1]; ++i){
					size_t c = col_idx[i];
					row_idx[col_ptr[c]] = r; 
					val[col_ptr[c]] = val_t[i];
					if(with_weights) weight[col_ptr[c]] = weight_t[i];	
					col_ptr[c]++;
				}
			}
			for(size_t c=cols; c>0; --c) col_ptr[c] = col_ptr[c-1];
			col_ptr[0] = 0;
		}
		size_t nnz_of_row(int i) const {return (row_ptr[i+1]-row_ptr[i]);}
		size_t nnz_of_col(int i) const {return (col_ptr[i+1]-col_ptr[i]);}
		double get_global_mean(){
			double sum=0;
			for(size_t i=0;i<nnz;++i) sum+=val[i];
			return sum/nnz;
		}
		void remove_bias(double bias=0){
			if(bias) {
				for(size_t i=0;i<nnz;++i) val[i]-=bias;
				for(size_t i=0;i<nnz;++i) val_t[i]-=bias;
			}
		}
		void free(void *ptr) {if(ptr) ::free(ptr);}
		~smat_t(){
			if(mem_alloc_by_me) {
				//puts("Warnning: Somebody just free me.");
				free(val); free(val_t);
				free(row_ptr);free(row_idx); 
				free(col_ptr);free(col_idx);
				if(with_weights) { free(weight); free(weight_t);}
			}
		}
		void clear_space() {
			free(val); free(val_t);
			free(row_ptr);free(row_idx); 
			free(col_ptr);free(col_idx);
			if(with_weights) { free(weight); free(weight_t);}
			mem_alloc_by_me = false;
			with_weights = false;

		}
		smat_t transpose(){
			smat_t mt;
			mt.cols = rows; mt.rows = cols; mt.nnz = nnz;
			mt.val = val_t; mt.val_t = val;
			mt.with_weights = with_weights;
			mt.weight = weight_t; mt.weight_t = weight;
			mt.col_ptr = row_ptr; mt.row_ptr = col_ptr;
			mt.col_idx = row_idx; mt.row_idx = col_idx;
			return mt;
		}
};


// row-major iterator
class smat_iterator_t: public entry_iterator_t{
	private:
		size_t *col_idx;
		size_t *row_ptr;
		double *val_t;
		double *weight_t;
		size_t	rows, cols, cur_idx, cur_row;
		bool with_weights;
	public:
		smat_iterator_t(const smat_t& M, int major = ROWMAJOR) {
			nnz = M.nnz;
			col_idx = (major == ROWMAJOR)? M.col_idx: M.row_idx;
			row_ptr = (major == ROWMAJOR)? M.row_ptr: M.col_ptr;
			val_t = (major == ROWMAJOR)? M.val_t: M.val;
			weight_t = (major == ROWMAJOR)? M.weight_t: M.weight; 
			with_weights = M.with_weights;
			rows = (major==ROWMAJOR)? M.rows: M.cols;
			cols = (major==ROWMAJOR)? M.cols: M.rows;
			cur_idx = cur_row = 0;
		}
		~smat_iterator_t() {}
		rate_t next() {
			int i = 1, j = 1;
			double v = 0;
			while (cur_idx >= row_ptr[cur_row+1]) ++cur_row;
			if (nnz > 0) --nnz;
			else fprintf(stderr,"Error: no more entry to iterate !!\n");
			rate_t ret(cur_row, col_idx[cur_idx], val_t[cur_idx], with_weights? weight_t[cur_idx]: 1.0);
			cur_idx++;
			return ret;
		}
};


// Test set format
class testset_t{
	public:
	size_t rows, cols, nnz;
	vector<rate_t> T;
	testset_t(): rows(0), cols(0), nnz(0){}
	inline rate_t& operator[](const size_t &idx) {return T[idx];}
	void load(size_t _rows, size_t _cols, size_t _nnz, const char *filename) {
		int r, c; 
		double v;
		rows = _rows; cols = _cols; nnz = _nnz;
		T = vector<rate_t>(nnz);
		FILE *fp = fopen(filename, "r");
		for(size_t idx = 0; idx < nnz; ++idx){
			fscanf(fp, "%d %d %lg", &r, &c, &v); 
			T[idx] = rate_t(r-1,c-1,v);
		}
		fclose(fp);
	}
	void load_from_iterator(size_t _rows, size_t _cols, size_t _nnz, entry_iterator_t* entry_it){ 
		rows =_rows,cols=_cols,nnz=_nnz;
		T = vector<rate_t>(nnz);
		for(size_t idx=0; idx < nnz; ++idx) 
			T[idx] = entry_it->next();
	}
	double get_global_mean(){
		double sum=0;
		for(size_t i=0; i<nnz; ++i) sum+=T[i].v;
		return sum/nnz;
	}
	void remove_bias(double bias=0){
		if(bias) for(size_t i=0; i<nnz; ++i) T[i].v-=bias;
	}
};

double init_testset_residual(testset_t &testset, mat_t &W, mat_t &H);

#endif

