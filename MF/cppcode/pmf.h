#ifndef _PMF_H_
#define _PMF_H_

#include "util.h"

class parameter {
	public:
		int k;
		int threads;
		int maxiter, maxinneriter;
		double lambda;
		double eps;						// for the fundec stop-cond in ccdr1
		double outereps;
		int do_predict, verbose;
		int do_nmf;  // non-negative matrix factorization
    int do_init;
    char* init_file;
		parameter() {
			k = 10;
			maxiter = 5;
			maxinneriter = 5;
			lambda = 0.1;
			threads = 4;
			eps = 1e-3;
			outereps = 1e-6;
			do_predict = 0;
			verbose = 0;
			do_nmf = 0;
      do_init = 1;
      init_file = NULL;
		}
    virtual ~parameter(){
      if(init_file != NULL)
        free(init_file);
      init_file = NULL;
    }
};

double polyMF_SS(smat_t &R, mat_t &W, mat_t &H, testset_t &T, parameter &param, int *totaliter = NULL, double *objlist = NULL, double *timelist = NULL);

#endif
