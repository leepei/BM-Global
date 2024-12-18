/* -------------------------------------------------------------------------- */
/* SetSpv mexFunction */
/* -------------------------------------------------------------------------- */

#include "mex.h"
#include <string.h>

/* set values to sparse matrix S */
void mexFunction(int nargout, mxArray *pargout [], int nargin, const mxArray *pargin [])
{
    double *Sval;
    double *v;
    double  LL;  
    unsigned long L;
    unsigned long k;
    
    if (nargin != 3)
        mexErrMsgTxt ("Usage:  SetSpv ( S, v, L ) from v to S") ;

    /* ---------------------------------------------------------------- */
    /* inputs */
    /* ---------------------------------------------------------------- */
    Sval = mxGetPr( pargin [0] );
    v    = mxGetPr( pargin [1] );
    LL   = mxGetScalar( pargin [2] );  
    L = (unsigned long) LL;
    
    /* ---------------------------------------------------------------- */
    /* output */
    /* ---------------------------------------------------------------- */
	memcpy(Sval, v, sizeof(Sval) * L);
    
    return;
}
