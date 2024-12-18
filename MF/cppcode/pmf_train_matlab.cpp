/*
[W,H,S,obj,pmftime,polyiter,objlist,timelist] = pmf_train_matlab(M,Mt,cmd);
or
[W,H,S,obj,pmftime,polyiter,objlist,timelist] = pmf_train_matlab(M,Mt,W,H,cmd);

Input:
M: sparse matrix for completion
Mt: transpose of M
W, H: initial guess for factorization: WH^T ~= M

Output:
S: residual
obj: final obj
pmftime: runniing time
polyiter: inner iter
objlist: list of objective values
timelist: list of running time
*/
	

#include "mex.h"
#include <omp.h>

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL
#include "util.h"
#include "pmf.h"
#include <cstring>

bool with_weights;

void exit_with_help()
{
	mexPrintf(
	"Usage: [W H,S,obj,time,iter, objlist, timelist] = pmf_train(R, Rt, W, H [, 'pmf_options'])\n"
	"       [W H,S,obj,time,iter, objlist, timelist] = pmf_train(R, Rt, [, 'pmf_options'])\n"
	"     R is an m-by-n sparse double matrix\n"
	"     Rt is the transpose of R\n"
	"     W is an m-by-k dense double matrix\n"
	"     H is an n-by-k dense double matrix\n"
	"     S is the residual matrix that represents W*H' - R in the observed entries\n"
	"     If W and H are given, they will be treated as the initial values,\n"
	"     and \"rank\" will equal to size(W,2).\n"
	"options:\n"
	"    -k rank : set the rank (default 10)\n"    
	"    -n threads : set the number of threads (default 4)\n"    
	"    -l lambda : set the regularization parameter lambda (default 0.1)\n"    
	"    -t max_iter: set the number of iterations (default 5)\n"    
	"    -T max_iter: set the number of inner iterations used in CCDR1 (default 5)\n"    
	"    -e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)\n"     
	"    -E epsilon2 : set outer termination criterion epsilon (default 1e-6)\n"     
	"    -p do_predict: do prediction or not (default 0)\n"    
	"    -q verbose: show information or not (default 0)\n"
	"    -N do_nmf: do nmf (default 0)\n"
	"    -i file_name: use file name as initialization\n"    
	);
}

// nrhs == 2 or 3 => pmf(R, Rt, [, 'pmf_options']);
// nrhs == 4 or 5 => pmf(R, Rt, W, H, [, 'pmf_options']);
parameter parse_command_line(int nrhs, const mxArray *prhs[])
{
	parameter param;   // default values have been set by the constructor 
	int i, argc = 1;
	int option_pos = -1;
	char cmd[CMD_LEN];
	char *argv[CMD_LEN/2];

	with_weights = false;

	if(nrhs < 1)
		return param;

	// put options in argv[]
	if(nrhs == 3) option_pos = 2;
	if(nrhs == 5) option_pos = 4;
	if(option_pos>0)
	{
		mxGetString(prhs[option_pos], cmd,  mxGetN(prhs[option_pos]) + 1);
		if((argv[argc] = strtok(cmd, " ")) != NULL)
			while((argv[++argc] = strtok(NULL, " ")) != NULL)
				;
	}

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 'k':
				param.k = atoi(argv[i]);
				break;

			case 'n':
				param.threads = atoi(argv[i]);
				break;

			case 'l':
				param.lambda = atof(argv[i]);
				break;

			case 't':
				param.maxiter = atoi(argv[i]);
				break;

			case 'T':
				param.maxinneriter = atoi(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				break;

			case 'E':
				param.outereps = atof(argv[i]);
				break;

			case 'p':
				param.do_predict = atoi(argv[i]);
				break;

			case 'q':
				param.verbose = atoi(argv[i]);
				break;

			case 'N':
				param.do_nmf = atoi(argv[i]) == 1? true : false;
				break;

			case 'i':
				param.init_file = strdup(argv[i]);
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	if (param.do_predict!=0)
		param.verbose = 1;

	if (nrhs > 3) {
		if(mxGetN(prhs[2]) != mxGetN(prhs[3]))
			mexPrintf("Dimensions of W and H do not match!\n");
		param.k = (int)mxGetN(prhs[2]);
	}
	
	return param;
}

static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[3] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[4] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[5] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[6] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[7] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

class mxSparse_iterator_t: public entry_iterator_t {
	private:
		mwIndex *ir_t, *jc_t;
		double *v_t;
		size_t	rows, cols, cur_idx, cur_row;
	public:
		mxSparse_iterator_t(const mxArray *M, const mxArray *Mt){
			rows = mxGetM(M); cols = mxGetN(M);
			nnz = *(mxGetJc(M) + cols);
			ir_t = mxGetIr(Mt); jc_t = mxGetJc(Mt); v_t = mxGetPr(Mt);
			cur_idx = cur_row = 0;
		}
		rate_t next() {
			int i = 1, j = 1;
			double v = 0;
			while (cur_idx >= jc_t[cur_row+1]) ++cur_row;
			if (nnz > 0) --nnz;
			else fprintf(stderr,"Error: no more entry to iterate !!\n");
			rate_t ret(cur_row, ir_t[cur_idx], v_t[cur_idx]);
			cur_idx++;
			return ret;
		}
};

/*
smat_t mxSparse_to_smat(const mxArray *M, const mxArray *Mt, smat_t &R) {
	size_t rows = mxGetM(M), cols = mxGetN(M), nnz = *(mxGetJc(M) + cols);
	mxSparse_iterator_t entry_it(M, Mt);
	R.load_from_iterator(rows, cols, nnz, &entry_it);
	return R;
}
*/

// convert matab dense matrix to column fmt
int mxDense_to_matCol(const mxArray *mxM, mat_t &M) {
	size_t rows = mxGetM(mxM), cols = mxGetN(mxM);
	double *val = mxGetPr(mxM);
	M.m = rows;
	M.n = cols;
	M.entries = new double[M.m * M.n];
	memcpy(M.entries, val, sizeof(M.entries) * M.m * M.n);
	return 0;
}

int matCol_to_mxDense(const mat_t &M, mxArray *mxM) {
	size_t cols = M.n, rows = M.m;
	double *val = mxGetPr(mxM);
	if(cols != mxGetN(mxM) || rows != mxGetM(mxM)) {
		mexPrintf("matCol_to_mxDense fails (dimensions do not match)\n");
		return -1;
	}
	memcpy(val, M.entries, sizeof(val) * cols * rows);
	return 0;
}


int run_algo(mxArray *plhs[], int nrhs, const mxArray *prhs[], parameter &param){
	mxArray *mxW, *mxH, *mxtime, *mxobjs;
	mxArray *mxS;
//	smat_t R;
	mat_t W, H;
	testset_t T;  // Dummy

	double time = omp_get_wtime();
//	mxSparse_to_smat(prhs[0], prhs[1], R);
	smat_t R((size_t) mxGetM(prhs[0]), (size_t) mxGetN(prhs[0]),(size_t *) mxGetIr(prhs[0]), (size_t *) mxGetJc(prhs[0]),mxGetPr(prhs[0]), (size_t *) mxGetIr(prhs[1]), (size_t *) mxGetJc(prhs[1]),mxGetPr(prhs[1]));

	// fix random seed to have same results for each run
	// (for random initialization)
	srand(1);
	srand48(0L);

	if(nrhs > 3) {
		mxDense_to_matCol(prhs[2], W);
		mxDense_to_matCol(prhs[3], H);
	} else {
		initial_col(W, param.k, R.rows);
		initial_col(H, param.k, R.cols);
	}

	int iter = 0;
	double *objlist = new double[param.maxiter];
	double *timelist = new double[param.maxiter];
	double obj = polyMF_SS(R, W, H, T, param, &iter, objlist, timelist);
	// Write back the result
	double time2 = omp_get_wtime();
	plhs[0] = mxW = mxCreateDoubleMatrix(W.m, W.n, mxREAL);
	plhs[1] = mxH = mxCreateDoubleMatrix(H.m, H.n, mxREAL);
	plhs[2] = mxS = mxCreateDoubleMatrix(R.nnz,1, mxREAL);//mxS is X - R
	plhs[3] = mxCreateDoubleScalar(obj);
	plhs[4] = mxCreateDoubleScalar(time2 - time);
	plhs[5] = mxCreateDoubleScalar((double)iter);
	memcpy(mxGetPr(mxS),R.val,sizeof(R.val) * R.nnz);

	if (iter > 0)
	{
		plhs[6] = mxobjs = mxCreateDoubleMatrix(iter, 1, mxREAL);
		plhs[7] = mxtime = mxCreateDoubleMatrix(iter, 1, mxREAL);
		memcpy(mxGetPr(mxobjs), objlist, sizeof(objlist) * iter);
		memcpy(mxGetPr(mxtime), timelist, sizeof(timelist) * iter);
	}
	else
	{
		plhs[6] = mxCreateDoubleMatrix(0, 0, mxREAL);
		plhs[7] = mxCreateDoubleMatrix(0, 0, mxREAL);
	}

	matCol_to_mxDense(W, mxW);
	matCol_to_mxDense(H, mxH);
	delete[] R.val;
	delete[] R.val_t;
	delete[] W.entries;
	delete[] H.entries;
	delete[] objlist;
	delete[] timelist;

	// Destroy matrix we allocated in this function
	return 0;
}

// Interface function of matlab
void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[] )
{
	parameter param;
	// fix random seed to have same results for each run
	// (for cross validation)
	srand(1);

	// Transform the input Matrix to libsvm format
	if(nrhs > 0 && nrhs < 6)
	{
		if ((!mxIsDouble(prhs[0]) || !mxIsSparse(prhs[0])) || !mxIsDouble(prhs[0]) || !mxIsSparse(prhs[0])) {
			mexPrintf("Error: matrix must be double and sparse\n");
			fake_answer(plhs);
			return;
		}

		param = parse_command_line(nrhs, prhs);
	//	if (param.maxiter > 0)
		run_algo(plhs, nrhs, prhs, param);
	}
	else {
		exit_with_help();
		fake_answer(plhs);
	}
}
