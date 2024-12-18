#include "util.h"
#include "pmf.h"
#include <iostream>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <climits>
#define kind dynamic,500
#ifdef __cplusplus
extern "C" {
#endif

extern int daxpy_(int *, double *, double *, int *, double *, int *);
extern double ddot_(int *, double *, int *, double *, int *);

#ifdef __cplusplus
}
#endif

#define __DEBUG__ 0
int findSubOpt(double r, double AA1, double RA, double BB1, double RB, double &alpha, double &beta) 
{

  // prepare paremeters
  double C = std::sqrt(AA1*BB1);
  double normA = RA/std::sqrt(AA1), normB = RB/std::sqrt(BB1);
  double delp = (normA+normB)/(2*C), deln = (normA-normB)/(2*C);
  double R = r/C;
  double delp2 = delp*delp;
  double deln2 = deln*deln;

  // handle the special cases
  int n_roots = 5;
  double L_min = 1e41;
  double z_min = 0, x_min = 0, y_min = 0;
  double meps = 1e-20;
  bool p_is_root = false, n_is_root = false;
  std::complex<double> z[5];
  if(deln2 < 1e-12 and delp2-4*(1-R) >= 0){
    p_is_root = true;
    z[--n_roots] = z[--n_roots] = 1.;
    double L = 1/2. + delp2/2 -1 +R;
    //fprintf(stderr, "z=1 L=%f\n", L);
    if(L_min > L){
      L_min = L;
      z_min = 1;
      x_min = -(normA+normB)/2;
      y_min = sqrt(delp2-4*(1-R))*C;
    }
  }
  if(delp2 < 1e-12 and deln2+4*(-1-R) >= 0){
    n_is_root = true;
    z[--n_roots] = z[--n_roots] = -1.;
    double L = 1/2. + deln2/2 -1 -R;
    //fprintf(stderr, "z=-1 L=%f\n", L);
    if(L_min > L){
      L_min = L;
      z_min = -1;
      y_min = (normA-normB)/2;
      x_min = sqrt(deln2+4*(-1-R))*C;
    }
  }
  if(p_is_root and n_is_root){
    z[--n_roots] = R;
    double L = R*R*( 1/2. + delp/((R+1)*(R+1)) + deln/((R-1)*(R-1)) );
    //fprintf(stderr, "z=%f L=%f\n", R, L);
    if(L_min > L){
      L_min = L;
      z_min = R;
      x_min = -(normA+normB)/(1+R);
      y_min = -(normA-normB)/(1-R);
    }
  }
  
  // init
  double sc = 1+2*std::abs(R+delp2+deln2);
  double eps = 1e-10*sc;
  std::complex<double> u(0.6, 0.8);
  z[0] = u*u*u*u*u*u*sc;
  z[1] = u*sc;
  z[2] = u*u*sc;
  z[3] = u*u*u*sc;
  z[4] = u*u*u*u*sc;
  
  // solve the remaining roots
  int iter = 0, max_iter = 100;
  if(n_roots == 0)
    max_iter = 0; // just skip

  double best_err = 0;
  double nt_start = omp_get_wtime();
  std::complex<double> zp[5], zn[5], f[5];
  for(; iter < max_iter; iter++) {
    if(__DEBUG__){
      std::cerr << "\n";
      std::cerr << z[0] << '\t' << z[1] << '\t' << z[2] << '\t' << z[3] << '\t' << z[4] << "\n";
    }

    zp[0] = z[0]+1., zn[0] = z[0]-1.;
    zp[0] *= zp[0],  zn[0] *= zn[0];
    zp[1] = z[1]+1., zn[1] = z[1]-1.;
    zp[1] *= zp[1],  zn[1] *= zn[1];
    zp[2] = z[2]+1., zn[2] = z[2]-1.;
    zp[2] *= zp[2],  zn[2] *= zn[2];
    zp[3] = z[3]+1., zn[3] = z[3]-1.;
    zp[3] *= zp[3],  zn[3] *= zn[3];
    zp[4] = z[4]+1., zn[4] = z[4]-1.;
    zp[4] *= zp[4],  zn[4] *= zn[4];

    if(p_is_root or n_is_root){
      if(p_is_root){
        f[0] = (z[0]-R)*zn[0] + deln2;
        f[1] = (z[1]-R)*zn[1] + deln2;
        f[2] = (z[2]-R)*zn[2] + deln2;
      }else{
        f[0] = (z[0]-R)*zp[0] - delp2;
        f[1] = (z[1]-R)*zp[1] - delp2;
        f[2] = (z[2]-R)*zp[2] - delp2;
      }
      z[0] = z[0] - f[0] / ((z[0]-z[1])*(z[0]-z[2]));
      z[1] = z[1] - f[1] / ((z[1]-z[0])*(z[1]-z[2]));
      z[2] = z[2] - f[2] / ((z[2]-z[0])*(z[2]-z[1]));
    }else{
      f[0] = ((z[0]-R)*zn[0] + deln2)*zp[0] - delp2*zn[0];
      f[1] = ((z[1]-R)*zn[1] + deln2)*zp[1] - delp2*zn[1];
      f[2] = ((z[2]-R)*zn[2] + deln2)*zp[2] - delp2*zn[2];
      f[3] = ((z[3]-R)*zn[3] + deln2)*zp[3] - delp2*zn[3];
      f[4] = ((z[4]-R)*zn[4] + deln2)*zp[4] - delp2*zn[4];

      z[0] = z[0] - f[0] / ((z[0]-z[1])*(z[0]-z[2])*(z[0]-z[3])*(z[0]-z[4]));
      z[1] = z[1] - f[1] / ((z[1]-z[0])*(z[1]-z[2])*(z[1]-z[3])*(z[1]-z[4]));
      z[2] = z[2] - f[2] / ((z[2]-z[0])*(z[2]-z[1])*(z[2]-z[3])*(z[2]-z[4]));
      z[3] = z[3] - f[3] / ((z[3]-z[0])*(z[3]-z[1])*(z[3]-z[2])*(z[3]-z[4]));
      z[4] = z[4] - f[4] / ((z[4]-z[0])*(z[4]-z[1])*(z[4]-z[2])*(z[4]-z[3]));
    }

    double max_err = max(abs(f[0]), abs(f[1]));
    max_err = max(max_err, abs(f[2]));
    max_err = max(max_err, abs(f[3]));
    max_err = max(max_err, abs(f[4]));
    if(max_err < eps){
      best_err = max_err;
      break;
    }
    if(__DEBUG__)
      std::cerr << f[0] << '\t' << f[1] << '\t' << f[2] << '\t' << f[3] << '\t' << f[4] << "\n";
  }
  if(iter == max_iter){
    std::cerr << R << " " << C << " " <<  delp2 << " " << deln2 << " " << eps << "\n";
    std::cerr << "Sub failed\n";
  }

  for(int i=0; i<n_roots; i++){
    double zi = std::real(z[i]);

    // there should be no zi=+-1 here.
    double x = -(normA+normB)/(zi+1);
    double y = -(normA-normB)/(zi-1);
    // recover zi from x,y
    // since the eq may not hold strictly
    double zz = (x*x-y*y)/(4*C*C)+R;

    double L;
    L = delp2/((zi+1)*(zi+1));
    L += deln2/((zi-1)*(zi-1));
    L *= zi*zi;
    L += zz*zz/2;

    //fprintf(stderr, "z=%.3g+%.3gi\terr=%f\tL=%f\n", zi, std::imag(z[i]), std::abs(f[i]), L);
    if(L_min > L){
      L_min = L;
      z_min = zi;
      x_min = x;
      y_min = y;
    }
  }

  //std::cerr << "choose " << z_min << "\n\n";
  alpha = (x_min+y_min)/2/std::sqrt(BB1);
  beta  = (x_min-y_min)/2/std::sqrt(AA1);

  // examine
  double r_alpha = alpha*beta*beta + r*beta + BB1*alpha + RB;
  double r_beta = alpha*alpha*beta + r*alpha + AA1*beta + RA;

  //std::cerr << " f_min = " << f_min << ", z = " << z_min << "\n";
/*  if(std::abs(r_alpha)>1e-4 or std::abs(r_beta)>1e-4){
    fprintf(stderr, "ERR: r=%f AA1=%f RA=%f BB1=%f RB=%f\n", r, AA1, RA, BB1, RB);
    std::cerr << "ERR: Dalpha = " << r_alpha << ", Dbeta = "  << r_beta << ", err = " << best_err << "\n";
    std::cerr << "ERR: choose " << z_min << " from\n";
    for(int i=0; i<n_roots; i++){
      std::cerr << z[i];
    }
    fprintf(stderr, "\n");
  }*/

  return iter;
}

double frand()
{
	return random()*1. / INT_MAX;
}

void swap_int(int *a, int i, int j)
{
	int t = a[i];
	a[i] = a[j];
	a[j] = t;
}

void permutation(int *a, int l)
{
	for(int i=0; i<l; i++){
		int j = (int)random() % (l-i);
		swap_int(a, i, j);
	}
}

inline double RankOneUpdate(const smat_t &R, const int i, double *v, const double lam, const double ui, double &diff){
	double g=0, h=lam;
	if(R.row_ptr[i+1]==R.row_ptr[i]) return 0;
	for(long idx=R.row_ptr[i]; idx < R.row_ptr[i+1]; ++idx) {
		int j = R.col_idx[idx];
		g += v[j]*R.val_t[idx];
		h += v[j]*v[j];
	}
	double newui = -g/h;
	double delta = newui - ui;
	diff += h*delta*delta;
	return newui;
}

inline void UpdateRating(smat_t &R, double *a, double *b, bool add)
{
  if(add){
#pragma omp parallel for schedule(kind) shared(a,b)
    for(int i=0; i<R.rows; i++){
      double ai = a[i];
      for(int idx=R.row_ptr[i]; idx<R.row_ptr[i+1]; idx++){
        R.val_t[idx] += ai * b[R.col_idx[idx]];
      }
    }
  }else{
#pragma omp parallel for schedule(kind) shared(a,b)
    for(int i=0; i<R.rows; i++){
      double ai = a[i];
      for(int idx=R.row_ptr[i]; idx<R.row_ptr[i+1]; idx++){
        R.val_t[idx] -= ai * b[R.col_idx[idx]];
      }
    }
  }
}

inline double UpdateRatingAndSum(smat_t &R, double *a, double *b, bool add) 
{
  double loss = 0;
  if(add){
#pragma omp parallel for schedule(kind) shared(a,b) reduction(+:loss)
    for(int i=0; i<R.rows; i++){
      double ai = a[i];
      for(int idx=R.row_ptr[i]; idx<R.row_ptr[i+1]; idx++){
        R.val_t[idx] += ai * b[R.col_idx[idx]];
        loss += R.val_t[idx]*R.val_t[idx];
      }
    }
  }else{
#pragma omp parallel for schedule(kind) shared(a,b) reduction(+:loss)
    for(int i=0; i<R.rows; i++){
      double ai = a[i];
      for(int idx=R.row_ptr[i]; idx<R.row_ptr[i+1]; idx++){
        R.val_t[idx] -= ai * b[R.col_idx[idx]];
        loss += R.val_t[idx]*R.val_t[idx];
      }
    }
  }
  return loss;
}

void solve_step(double W, double XB, double XA, double r, double BB1, double AA1, double RB, double RA, double &diff, double &alpha, double &beta, int &n_kerner)
{
  //fprintf(stderr, "W=%g XB=%g XA=%g r=%f BB1=%g AA1=%g RB=%g RA=%g\n", W, XB, XA, r, BB1, AA1, RB, RA);
  if(std::abs(W) < 1e-18){
    if(std::abs(BB1) > 1e-18)
      alpha = -(RB+beta*(r+beta*XA))/(BB1+beta*(XB+beta*W));
    else
      alpha = 0;
    if(std::abs(AA1) > 1e-18)
      beta = -(RA+alpha*(r+alpha*XB))/(AA1+alpha*(XA+alpha*W));
    else
      beta = 0;
    double new_L = alpha*beta*(W*alpha*beta/2+XB*alpha+XA*beta+r)+alpha*(BB1*alpha/2+RB) + beta*(AA1*beta/2+RA);
    diff = (0-new_L)*2;
  }else{
//  fprintf(stderr, "kerner\n");
    XB /= W, XA /= W, r /= W, BB1 /= W, AA1 /= W, RB /= W, RA /= W;
    double r_ = r-2*XA*XB;
    double BB1_ = BB1-XB*XB;
    double RB_ = RB+2*XA*XB*XB-r*XB-XA*BB1;
    double AA1_ = AA1-XA*XA;
    double RA_ = RA+2*XA*XA*XB-r*XA-XB*AA1;

    double alpha_=XA, beta_=XB;
    double old_L = alpha_*(beta_*(alpha_*beta_/2+r_)+BB1_/2*alpha_+RB_)+beta_*(AA1_*beta_/2+RA_);
    n_kerner += findSubOpt(r_, AA1_, RA_, BB1_, RB_, alpha_, beta_);
    //if(oiter>=1)
    //  alpha_ = 1+XA, beta_=1+XB;
    double new_L = alpha_*(beta_*(alpha_*beta_/2+r_)+BB1_/2*alpha_+RB_)+beta_*(AA1_*beta_/2+RA_);
    diff = W*(old_L-new_L)*2;
    alpha = alpha_ - XA;
    beta = beta_ - XB;
    alpha_ = 1+XA, beta_=1+XB;
  }
}

void validate_res(const double *S, const double *St, smat_t &R, mat_t &A, mat_t &B, double &reg, double &loss)
{
  int K = A.n;
  int M = R.rows;
  int N = R.cols;
  printf("M = %d, N = %d, K = %d\n",M,N,K);
  reg = 0, loss = 0;
  for(int i=0; i<M; i++){
    for(int idx=R.row_ptr[i]; idx<R.row_ptr[i+1]; idx++){
      int j = R.col_idx[idx];
      double rij = R.val_t[idx];
      double Sij = S[idx];
      for(int kk=0; kk<K; kk++){
        Sij -= A.entries[kk*M + i]*B.entries[kk*N + j];
      }
      Sij = -Sij;
//      if(std::abs(Sij-rij) > 1e-7){
 //       fprintf(stderr, "err (%d,%d) Sij=%f rij=%f\n", i, j, Sij, rij);
  //    }
      loss += rij*rij;
    }
  }
  for(int j=0; j<N; j++){
    for(int idx=R.col_ptr[j]; idx<R.col_ptr[j+1]; idx++){
      int i = R.row_idx[idx];
      double rij = R.val[idx];
      double Sij = St[idx];
      for(int kk=0; kk<K; kk++){
        Sij -= A.entries[kk*M+i]*B.entries[kk*N+j];
      }
      Sij = -Sij;
//      if(std::abs(Sij-rij) > 1e-7){
//        fprintf(stderr, "err_t (%d,%d) Sij=%f rij=%f\n", i, j, Sij, rij);
  //    }
    }
  }
  for(int kk=0; kk<K; kk++){
    for(int i=0; i<M; i++)
      reg += A.entries[kk*M+i]*A.entries[kk*M+i];
    for(int j=0; j<N; j++)
      reg += B.entries[kk*N+j]*B.entries[kk*N+j];
  }
}

double polyMF_SS(smat_t &R, mat_t &A, mat_t &B, testset_t &T, parameter &param, int *totaliter, double *objlist, double *timelist)
{
	int inc = 1;
	int num_threads_old = omp_get_num_threads();
	omp_set_num_threads(param.threads);
	double run_time=0;
	double inner_time=0., inner_start;
	double kerner_time=0., kerner_start;
	int n_kerner = 0, n_update = 0;
	double mone = -1;

//  fprintf(stderr, "r.rows=%ld r.cols=%ld A=%ld B=%ld k=%ld\n", R.rows, R.cols, A.m*A.n, B.m*B.n, param.k);
  int maxiter   = param.maxiter;
  int maxinneriter = param.maxinneriter;
  int K         = param.k;
  int M         = R.rows;
  int N         = R.cols;
  double lam    = param.lambda;
  double eps    = param.eps;
  smat_t Rt = R.transpose();

  double obj = 0., old_obj = 0.;
  double reg=0, loss;

  // init
//  if(param.do_init){
//	  memset(B.entries, 0, sizeof(double)*K*N);
//    for(int k=0; k<K; k++){
      //fill_n(A[k].begin(), M, 0.);
      //fill_n(B[k].begin(), N, 0.);
//    }
//  }

  for(int k=0; k<K; k++){
    UpdateRating(R, A.entries + k * M, B.entries + k * N, false);
  }
  for(int k=0; k<K; k++){
    UpdateRating(Rt, B.entries + k * N, A.entries + k * M, false);
  }
  int Asize = A.m * A.n;
  int Bsize = B.m * B.n;
  reg = 0;
  for (int i=0;i<Asize;i++)
	  reg += A.entries[i] * A.entries[i];
  for (int i=0;i<Bsize;i++)
	  reg += B.entries[i] * B.entries[i];
//  reg = ddot_(&Asize, A.entries, &inc, A.entries, &inc) + ddot_(&Bsize, B.entries, &inc, B.entries, &inc);
  loss = 0;
#pragma omp parallel for reduction(+:loss)
  for(int idx=0; idx<R.nnz; idx++){
    // our definition is ai*bj-Sij
    R.val_t[idx] *= -1; 
    R.val[idx] *= -1; 
    loss += R.val_t[idx] * R.val_t[idx];
  }
  old_obj = obj = lam*reg+loss;
  double outer_old_obj = old_obj;

  if(T.nnz!=0 and param.do_predict)
    init_testset_residual(T, A, B);


  vec_t u(M), v(N);
  vec_t old_a(M), old_b(N);
  int *perm = new int[K];
  for (int i=0;i<K;i++)
	  perm[i] = i;
  int oiter;
  double rmse = 0;
  for(oiter=0; oiter<maxiter; oiter++) {
	  double fundec_max = 0;
		permutation(perm,K);
		
    for(int k=0; k < K; k++) {
		int idx = perm[k];

      inner_time = 0;
      inner_start = omp_get_wtime();

      // We simply all the k in the notation
      double *a = A.entries + (idx*M), *b = B.entries + (idx*N);
	  memcpy(old_a.data(), a, sizeof(double) * M);
	  memcpy(old_b.data(), b, sizeof(double) * N);

      //old_a = a, old_b = b;
      
      kerner_start = omp_get_wtime();

      UpdateRating(R, a, b, false);
      UpdateRating(Rt, b, a, false);

      double diff, improve_L;
      for(int iter=0; iter<maxinneriter; iter++) {
        diff = 0;
#pragma omp parallel for schedule(kind) shared(a,b) reduction(+:diff)
        for(int j = 0; j < N; j++)
          b[j] = RankOneUpdate(Rt, j, a, lam, b[j], diff);


#pragma omp parallel for schedule(kind) shared(a,b) reduction(+:diff)
        for(int i = 0; i < M; i++)
          a[i] = RankOneUpdate(R, i, b, lam, a[i], diff);


        if((diff < fundec_max*eps))  {
                break; 
        }

        if(!(oiter==1 && k == 0 && iter==1))
                fundec_max = max(fundec_max, diff);
      }
      
      kerner_time = 0.;
      double kerner_start = omp_get_wtime();

/*	  memcpy(u.data(), a, sizeof(a) * M);
	  memcpy(v.data(), b, sizeof(b) * N);
	  daxpy_(&M, &mone, old_a.data(), &inc, u.data(), &inc);
	  daxpy_(&N, &mone, old_b.data(), &inc, v.data(), &inc);*/
	  for (int i=0;i<M;i++)
		  u[i] = a[i] - old_a[i];
	  for (int i=0;i<N;i++)
		  v[i] = b[i] - old_b[i];

      double W=0., XB=0., XA=0., r=0., BB1=0., AA1=0., RB=0., RA=0.;
	  for (int i=0;i<M;i++)
	  {
		  BB1 += u[i] * u[i];
		  RB += a[i] * u[i];
	  }
	  for (int i=0;i<N;i++)
	  {
		  AA1 += v[i] * v[i];
		  RA += b[i] * v[i];
	  }
//	  BB1 = ddot_(&M, u.data(), &inc, u.data(), &inc);
//	  RB = ddot_(&M, a, &inc, u.data(), &inc);
//	  AA1 = ddot_(&N, v.data(), &inc, v.data(), &inc);
//	  RA = ddot_(&N, b, &inc, v.data(), &inc);
	  BB1 *= lam;
	  AA1 *= lam;
	  RA *= lam;
	  RB *= lam;

#pragma omp parallel for schedule(kind) shared(a,b,u,v) reduction(+:W,XB,XA,r,BB1,AA1,RB,RA)
      for(int i=0; i<M; i++){
        for(int idx=R.row_ptr[i]; idx<R.row_ptr[i+1]; idx++){
          int j = R.col_idx[idx];
          double rij = R.val_t[idx]+a[i]*b[j];
          double uivj = u[i]*v[j];
          double uibj = u[i]*b[j];
          double vjai = v[j]*a[i];
          W += uivj*uivj;
          XB += uivj*uibj;
          XA += uivj*vjai;
          r += uivj*rij + uibj*vjai;
          BB1 += uibj*uibj;
          AA1 += vjai*vjai;
          RB += rij*uibj;
          RA += rij*vjai;
        }
      }
      
      double alpha, beta;
      solve_step(W, XB, XA, r, BB1, AA1, RB, RA, diff, alpha, beta, n_kerner);
      if(oiter==0)
        alpha = beta = diff = 0;
//      if(diff < 0)
  //      fprintf(stderr, "ERR in line search: diff = %g\n", diff);
      improve_L = diff;
      //fprintf(stderr, "diff_L=%f improve=%f alpha=%f beta=%f\n", W*(old_L-new_L)*2, W*(ccd_L-new_L)*2, alpha, beta);
      n_update ++;

      kerner_time = omp_get_wtime() - kerner_start;
	  for (int i=0;i<M;i++)
		  a[i] += alpha * u[i];
	  for (int i=0;i<N;i++)
		  b[i] += beta * v[i];
//	  daxpy_(&M, &alpha, u.data(), &inc, a, &inc);
//	  daxpy_(&N, &beta, v.data(), &inc, b, &inc);

      loss = UpdateRatingAndSum(R, a, b, true);
      UpdateRating(Rt, b, a, true);

      inner_time = omp_get_wtime() - inner_start;
      run_time += inner_time;
	  for (int i=0;i<M;i++)
		  reg += a[i] * a[i] - old_a[i] * old_a[i];
	  for (int i=0;i<N;i++)
		  reg += b[i] * b[i] - old_b[i] * old_b[i];
//	  reg += ddot_(&M, a, &inc, a, &inc) + ddot_(&N, b, &inc, b, &inc) - ddot_(&M, old_a.data(), &inc, old_a.data(), &inc) - ddot_(&N, old_b.data(), &inc, old_b.data(), &inc);

      obj = lam*reg + loss;

#ifdef TRUE_OBJ
//      validate_res(S, St, R, A, B, reg, loss);
#endif



	  /*
      if(param.verbose)
	  {
		  fprintf(stderr, "\t\titer %d rank %d time %.10g obj %.16g diff %.5g kerner %2.2f%% nker %.2f improve %.5g alpha %.5g beta %.5g ",
				  oiter+1,idx+1, run_time, obj, old_obj - obj, kerner_time/inner_time*100, n_kerner*1./n_update, improve_L, alpha, beta);
		  if(T.nnz!=0 and param.do_predict){ 
			  if(param.verbose)
				  fprintf(stderr, "rmse %.10g", calrmse_r1(T, a, b, old_a.data(), old_b.data())); 
		  }
		  fprintf(stderr, "\n");
	  }
	  */
      old_obj = obj;
	  if(T.nnz!=0 and param.do_predict)
		  rmse = calrmse_r1(T, a, b, old_a.data(), old_b.data());
    }
	if (timelist != NULL)
		timelist[oiter] = run_time;
	if (objlist != NULL)
		objlist[oiter] = obj / 2;
	if(param.verbose)
	{
		printf("\t\titer %d time %.10g obj %.16g diff %.5g\n",
				oiter+1, run_time, obj/2, outer_old_obj - obj);
		if(T.nnz!=0 and param.do_predict)
			printf("rmse %.5g\n",rmse);
	}

	if ((outer_old_obj - obj) / obj < param.outereps && oiter > 0)
		break;
	else
		outer_old_obj = obj;
  }
  if (totaliter != NULL)
	  *totaliter = oiter;
	omp_set_num_threads(num_threads_old);
	return obj/2;
}
