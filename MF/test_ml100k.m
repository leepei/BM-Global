clc; clear; close all; 
warning('off'); 
rng('default');
startup;
load('ml100k.mat');

param.lambda = 15;
param.eps = 1e-12;
[x,info] = MF(M, param, Mtest);
