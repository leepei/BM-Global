# BM-Global: A Highly-Efficient Solver for Nuclear-Norm Regularized Matrix Optimization Problems

This repository contains a MATLAB implementation of the BM-Global algorithm for nuclear-norm regularized matrix optimization problems.  
Details of the algorithm can be found in the following paper:  
> Ching-pei Lee, Ling Liang, Tianyun Tang, and Kim-Chuan Toh, [*Accelerating Nuclear-Norm Regularized Low-Rank Matrix Optimization Through Burer-Monteiro Decomposition*](https://jmlr.org/papers/v25/23-0049.html). Journal of Machine Learning Research, 2024.

This repository includes implementations for the nuclear-norm-regularized matrix completion problem and the reformulated nuclear-norm-regularized QSDP problem discussed and evaluated in the above paper.

The subproblem solver for the matrix completion problem is the PolyMF-SS algorithm, as presented in the following paper (implementation provided by its first author):  
> Po-Wei Wang, Chun-Liang Li, and J. Kolter, *Polynomial Optimization Methods for Matrix Factorization*. Proceedings of the AAAI Conference on Artificial Intelligence, 2017.

The subproblem solver for the QSDP problem is the trust-region algorithm available in the [Manopt](https://github.com/NicolasBoumal/manopt) package.

## Installation

To compile the code, you will need MATLAB and GCC (or another C++ compiler compatible with MATLAB).  

The implementation for the matrix completion problem is located in the [MF](https://github.com/leepei/BM-Global/tree/main/MF) folder.  
To compile the code, navigate to this folder in MATLAB and run `install.m`.

The implementation for the QSDP problem is located in the [EDME](https://www.github.com/leepei/BM-Global/tree/main/EDME) folder.  
To run the code, download the Manopt package and extract it into the `EDME/manopt` folder.  
If the precompiled files are not compatible with your system, navigate to `EDME/solvers/mexfun/` and run `make.m` to recompile the necessary C++ code.

## Getting Started

To use the matrix completion implementation, run `MF/startup.m` first.  
A script for testing the package on the ml100k dataset is provided in `MF/test_ml100k.m`.

For the QSDP implementation, a script for testing the package on a small RKE dataset is provided in `EDME/test_rke.m`.

## Alternative C++ Compilers

If you are using a C++ compiler other than `g++`, you can modify the compilation scripts in `MF/cppcode/make.m` and `EDME/solvers/mexfun/make.m` to specify the compiler. After modifying the scripts, run them to recompile the code.
