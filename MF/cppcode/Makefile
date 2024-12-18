CXX=g++-7
CXXFLAGS=-fopenmp -fPIC -pipe -O3
LIBS=-llapack -lblas -lgfortran -lm

all: omp-pmf-train omp-pmf-predict

omp-pmf-train: pmf-train.cpp pmf.h util.o kerner.o
	${CXX} ${CXXFLAGS} -o omp-pmf-train pmf-train.cpp util.o kerner.o $(LIBS)

omp-pmf-predict: pmf-predict.cpp pmf.h util.o
	${CXX} ${CXXFLAGS} -o omp-pmf-predict pmf-predict.cpp  util.o 

util.o: util.h util.cpp
	${CXX} ${CXXFLAGS} -c -o util.o util.cpp $(LIBS)
kerner.o: kerner.cpp
	${CXX} ${CXXFLAGS} -c -o kerner.o kerner.cpp $(LIBS)

clean:
	rm -rf  omp-pmf* *.o *.model *.mexa64

