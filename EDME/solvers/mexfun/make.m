% This make.m is for MATLAB under Unix

% Add -largeArrayDims on 64-bit machines of MATLAB
%Using g++-7 is basically to downgrade to a version that is compatible with MATLAB. g++-9 fails. If you don't have issues here, you can remove the part of CXX='g++-7'
mex -largeArrayDims CFLAGS="\$CFLAGS -O3 -fopenmp -D MATLAB" LDFLAGS="\$LDFLAGS -fopenmp -D MATLAB" COMPFLAGS="\$COMPFLAGS -openmp -D MATLAB" -cxx CXX='g++-7' mexhessedme_manopt.c
mex -largeArrayDims CFLAGS="\$CFLAGS -O3 -D MATLAB" LDFLAGS="\$LDFLAGS -D MATLAB" COMPFLAGS="\$COMPFLAGS -D MATLAB" -cxx CXX='g++-7' mexgradedme_manopt.c
mex -largeArrayDims CFLAGS="\$CFLAGS -O3 -fopenmp -D MATLAB" LDFLAGS="\$LDFLAGS -fopenmp -D MATLAB" COMPFLAGS="\$COMPFLAGS -openmp -D MATLAB" -cxx CXX='g++-7' ehess_EDME.c
