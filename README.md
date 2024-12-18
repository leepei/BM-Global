# Accelerating Nuclear-norm Regularized Low-rank Matrix Optimization Through Burer-Monteiro Decomposition

This repository is a MATlAB implementation of the BM-Global algorithm for nuclear-norm regularized matrix optimization problems.
Details of the algorithm can be found in the following paper:
>  Ching-pei Lee, Ling Liang, Tianyun Tang, and Kim-Chuan Toh, *Accelerating Nuclear-norm Regularized Low-rank Matrix Optimization Through Burer-Monteiro Decomposition*. Journal
of Machine Learning Research, 2024. [[Link](https://arxiv.org/abs/2204.14067)]

In this repository, we include our implementation for the nuclear-norm-regularized matrix completion problem and for the reformulated nuclear-norm-regularized QSDP problem discussed and experimented in the above paper.

The subproblem solver for the matrix completion problem is the PolyMF-SS algorithm of the following paper, provided by its first author:
> Po-Wei Wang, Chun-Liang Li, and J. Kolter. *Polynomial optimization methods for matrix factorization*. Proceedings of the AAAI Conference on Artificial Intelligence. 2017.

The subproblem solver for the QSDP problem is the trust-region algorithm in the package [Manopt](https://github.com/NicolasBoumal/manopt).

## Installation
To compile the code, you will need to install MATLAB and gcc (or other compiler for C++ compatible with MATLAB).
Implementation for the matrix completion problem is contained in the folder [MF](https://www.github.com/leepei/BM-Global/MF).
To compile the code, enter this folder in MATLAB and run install.m.

The implementation ofr the QSDP problem is contained in the folder [EDME](https://www.github.com/leepei/BM-Global/EDME).
To run the code, please download manopt first and extract it to the folder EDME/manopt.
You might need to enter EDME/solvers/mexfun/ and run make.m to recompile some C++ code if the precompiled files are not compatible with your system.

## Getting started
To use the matrix completion implementation, please run MF/startup.m first.
A script for testing our package on the ml100k data set is provided in MF/test_ml100k.m.

For the QSDP implementation,
our script for testing our package on a small RKE data set is provided in EDME/test_rke.m.

## Alternative compilers for C++
If you are using compilers for C++ other than g++, you can modify the code in MF/cppcode/make.m and EDME/solvers/mexfun/make.m to your compiler, and then run those two scripts to recompile the code.
