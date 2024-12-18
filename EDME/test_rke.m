clc;
close all;
clear;
warning('off');
rng(0);

addpath('../main');
addpath(genpath(pwd));


filelist{1} = 'proteinDisMatrix'; %%630 5e0
filelist{2} = 'catcortex'; %%65 5e-1
filelist{3} = 'BrainMRI'; %%124, 13X14 5e-1
filelist{4} = 'CoilDelftDiff'; %% 288 1e0
filelist{5} = 'coildelftsame'; %% 288 1e0
filelist{6} = 'delftgestures'; %% 1500 3e1
filelist{7} = 'flowcytodis'; %% 612, 1X4 2e0
filelist{8} = 'aviris-clear'; %% 145, 1X200 1e5
filelist{9} = 'newsgroups'; %% 600 5e1
filelist{10} = 'prodom'; %% 2604 1e-1
filelist{11} = 'protein'; %% 213 1e0
filelist{12} = 'WoodyPlants50'; %% 791 8e0
filelist{13} = 'zongker'; %% 2000, 5e0
filelist{14} = 'DelftPedestrians'; %% 689 1e1
filelist{15} = 'Chickenpieces-5-45'; %% 446 1e0
filelist{16} = 'Chickenpieces-5-120'; %% 446 1e0


stoptol = 1e-6;
useacc = 1;
alpfix = 1;

% maxNumCompThreads(4);

load('catcortex.mat');

if exist('a','var')
	if iscell(a)
		DD = a{1,1}.data;
	else
		DD = a.data;
		DD = DD(:,1);
		n = sqrt(length(DD));
		DD = reshape(DD, [n,n]);
	end
elseif exist('s', 'var')
	DD = s.data;
elseif exist('d', 'var') && iscell(d)
	DD = d{1,1}.data;
elseif  exist('d', 'var') && isstruct(d)
	DD = d.data;
elseif k == 15
	DD = 1-d;
else
	DD = D;
end    
	
DD = DD / max(DD(:));
DD = 0.5*(DD + DD');
D = triu(DD, 1);
D = D .* D;
n = size(D,1);
sparseLevel = 0.1;
if sparseLevel < 0.5
	Dtmp = sprand(n, n, sparseLevel);
	Dtmp = triu(Dtmp, 1);
	[idxI, idxJ] = find(Dtmp);
else
	[idxI, idxJ] = find(D);
end

W = ones(n,n)/2;
[blk, At, b, C, dd, ww] = genblk_new(idxI, idxJ, W, D);

%%
lambda = 1e-1*sqrt(n); %min(1, 5e1/sqrt(sum(b)));
maxrank = 30;min(100, floor(n*0.15));
Lip_const = 1e2;
C = C*lambda - smat(blk, At*b);
idx3d = [n, n-1, n-2];

options.maxiter = 500;
options.stoptol = stoptol;
options.useacc = useacc;
options.lipconst = Lip_const;
options.alpfix = alpfix;
options.maxrank = maxrank;
options.maxitermanopt = 20;
options.printmanopt = 0;
options.maxiterAPG = 100;
[v2, info2] = stride_edme_new(idxI, idxJ, ww, dd, lambda, n, maxrank, options);
[U2, S2] = eig(v2*v2');
d2 = diag(S2);
A2 = U2(:,idx3d)*diag(d2(idx3d).^0.5);
relkkt2 = rel_kkt_full(blk, At, C, v2*v2');
errRp2 = norm(v2*(v2'*ones(n,1))) / (1 + norm(v2,'fro')^2);
