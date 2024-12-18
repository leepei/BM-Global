% This make.m is for MATLAB and OCTAVE under Windows, Mac, and Unix

	Type = ver;
	% This part is for OCTAVE
	if(strcmp(Type(1).Name, 'Octave') == 1)
		system('make kerner.o util.o');
		mex -lgomp -llapack_atlas -lf77blas -lcblas -latlas -lgfortran pmf_train_matlab.cpp kerner.o util.o
		
	% This part is for MATLAB
	
	% Add -largeArrayDims on 64-bit machines of MATLAB
	else
		system('make kerner.o util.o');
	%Using g++-7 is basically to downgrade to a version that is compatible with MATLAB. g++-9 fails. If you don't have issues here, you can remove the part of CXX='g++-7'
		mex -largeArrayDims CFLAGS="\$CFLAGS -O3 -fopenmp -D MATLAB" LDFLAGS="\$LDFLAGS -fopenmp -D MATLAB" COMPFLAGS="\$COMPFLAGS -openmp -D MATLAB" -cxx CXX='g++-7' partXY.c
		mex -largeArrayDims CFLAGS="\$CFLAGS -O3 -fopenmp -D MATLAB" LDFLAGS="\$LDFLAGS -fopenmp -D MATLAB" COMPFLAGS="\$COMPFLAGS -openmp -D MATLAB" -cxx CXX='g++-7' setSval.c
		mex -largeArrayDims CFLAGS="\$CFLAGS -O3 -fopenmp -D MATLAB" LDFLAGS="\$LDFLAGS -fopenmp -D MATLAB" COMPFLAGS="\$COMPFLAGS -openmp -D MATLAB" -cxx -lgomp CXX='g++-7' pmf_train_matlab.cpp kerner.cpp util.cpp
	end
