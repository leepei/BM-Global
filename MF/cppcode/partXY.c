/* -------------------------------------------------------------------------- */
/* partXY_mex mexFunction */
/* -------------------------------------------------------------------------- */

#include "mex.h"
#include <omp.h>

/* compute a part of X*Y^T */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin  [ ]
)
{
	int threads;
	int num_threads_old = omp_get_num_threads();
	if (nargin == 6)
	{
		threads = int(mxGetScalar(pargin[5]));
		omp_set_num_threads(threads);
	}
    double *Xt, *Y, *Z, *I, *J, *v, LL;
    ptrdiff_t m, n, r, L, p, ir, jr, k;
    ptrdiff_t inc = 1;

    if (nargin < 5 || nargout > 1)
        mexErrMsgTxt ("Usage: v = partXY (Xt, Y, I, J, L, threads)") ;

    /* ---------------------------------------------------------------- */
    /* inputs */
    /* ---------------------------------------------------------------- */
    
    Xt = mxGetPr( pargin [0] );     // r x m
    Y  = mxGetPr( pargin [1] );     // r x n
    I  = mxGetPr( pargin [2] );     // row position
    J  = mxGetPr( pargin [3] );     // col position
    LL = mxGetScalar( pargin [4] ); // num of obvs
    L = (ptrdiff_t) LL;
    m  = mxGetN( pargin [0] );
    n  = mxGetN( pargin [1] );
    r  = mxGetM( pargin [0] ); 
    if ( r != mxGetM( pargin [1] ))
        mexErrMsgTxt ("rows of Xt must be equal to rows of Y") ;
    if ( r > m || r > n )
        mexErrMsgTxt ("rank must be r <= min(m,n)") ;
    
    /* ---------------------------------------------------------------- */
    /* output */
    /* ---------------------------------------------------------------- */

    pargout [0] = mxCreateDoubleMatrix(L,1, mxREAL);
    v = mxGetPr( pargout [0] );
    
    /* C array indices start from 0 */
#pragma omp parallel for schedule(dynamic,256)
    for (p = 0; p < L; p++) {
        ir = ( I[p] - 1 ) * r;
        jr = ( J[p] - 1 ) * r;
        v[p] = 0;
        for (k = 0; k < r; k++)
            v[p] += Xt[ ir + k ] * Y[ jr + k ];
    }
    
    return;
}

