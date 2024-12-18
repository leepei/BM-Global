function [x,info] = MF(M, param, Mtest)
%Problem formulation: min_{X in R^{m*n}} 1/2 ||P_{Omega}(X - M)||_F^2 + lambda ||X||_*
%
%Input parameters from param:
%	W,H: optional warmstart matrices that serve as the initial guess for X = W * H';
%	lambda: the weight for the nuclear norm regularizer (default 1)
%	profile: print profiling in the SVD and BM parts (default 0)
%	step: initial step size at each iteration (default 1)
%	eps: stopping condition (default 1e-6)
%	maxiter: max #iter of the whole algorithm (default 100)
%	exponent: the stopping condition in the BM step is tol^exponent (default 2)
%	base_iter: the default #epochs of each BM step (default 2)
%	k: initial rank (default min(m,n,#threads))
%	threads: number of computing threads/cores (default is the default of platform)
%	svditer: max_iter in SVDs (default 5)
%	maxk: max rank of W and H (default m)
%	pg_iter: run #pg_iters between two BM steps (default 1: alternate between PG and BM)
%   maxtime: max running time (in seconds)
%   obj: stop when objective is no greater than this value
%
%M: training data matrix
%Mtest: testing matrix
%
%Output:
%	x: structured as x.W and x.H, such that x = W*H'
%	info: structure for all output information
%		obj: the final objective value
%		time: total running time
%		objlist: the whole list of objective values, including change oevr epochs in BM steps
%		timelist: the whole list of elapsed time, including change oevr epochs in BM steps
%		outerobjlist: objective values at the end of each outer loop
%		outertimelist: elapsed time  at the end of each outer loop
%		proxiters: the list of iterations (in timelist and objlist) that a PG step is conducted
%				Note that proxiters record the places at where we finished 1
%				proximal gradient step (so time cost should be between
%				timelist(proxiters(i)-1) and timelist(proxiters(i)))
%		kSVDlist: the rank of the input to warmstart SVD at those points indicated by proxiters
%		kMFlist: the rank used in BM steps
%				(kMFlist(i+1) <= kSVDlist(i) because during SVD we might truncate
%				out some singular vectors, and the proximal step further truncates
%				some of them)
%		backtrackinglist: number of backtracking steps when linesearch is used
%		stepsizelist: the corresponding final accepted step sizes
%

	delta = .99;
	step_lower_bound = 1e-10;%Terminate backtracking when step size smaller than this value to prevent numerical error


	interchanged = 0;%A flag to record whether we interchange M and M' to satisfy m <= n
	[m,n] = size(M);
	if (m <= n)
		Mt = M';
	else
		interchanged = 1;
		Mt = M;
		M = Mt';
		ntmp = n;
		n = m;
		m = ntmp;
	end

	noise_shrink = 10;
	eta = 2;
	sigma = 1e-4; %Fixed parameters
	rng(2022);%For reproducing exps and comparing approaches; Results still change though, probably due to the C++ part doesn't have a fixed random seed
	eps = 1e-6;%Stopping condition: stop when objective change in 1 prox grad step <= eps.
	maxiter = 100;
	lambda = 1;
	profile = 1;
	stop_flag = 0;

	pg_iter = 1;%#PG iterations between 2 BM steps
	base_iter = 2;%#max_iter for MF subproblem solver; will increase when rank becomes stable
	exponent = 2;%MF subproblem solver stopping condition = obj change in 1 epoch <= tol^exponent, where tol = obj_change_from_prox_grad/new_obj
	svditer = 5;%Number of max iters of lmSVD
	filtereps0 = 1e-12;%threshold for removing redundant columns. Value small (not removing many) when rank is still increasing
	filtereps1 = 1e-4;%Switch to this one once the rank becomes stable
	maxtime_flag = 0;
	maxk = m;

	threads = maxNumCompThreads;
	test_flag = 0;
	k = threads; %Idea: fully parallelization starting from the 1st iteration. Smaller k means some cores are idle, while larger k means additional work/time
	alpha = 1;
	step0 = 1/alpha;
	tol_off = 0;%This flag is turned on when we found that rank is fixed but tolerance is too loose such that subproblem solver terminates before reaching the max iters allowed

	objflag = 0;
	if (nargin >= 2)
		if (isfield(param,'maxiter'))
			maxiter = param.maxiter;
		end
		if (isfield(param,'eps'))
			eps = param.eps;
		end
		if (isfield(param,'lambda'))
			lambda = param.lambda;
		end
		if (isfield(param,'exponent'))
			exponent = param.exponent;
		end
		if (isfield(param,'base_iter'))
			base_iter = param.base_iter;
		end
		if (isfield(param,'k'))
			k = param.k;
		end
		if (isfield(param,'threads'))
			threads = param.threads;
			maxNumCompThreads(threads);
		end
		if (isfield(param,'svditer'))
			svditer = param.svditer;
		end
		if (isfield(param,'maxk'))
			maxk = param.maxk;
		end
		if (isfield(param,'pg_iter'))
			pg_iter = param.pg_iter;
		end
		if (isfield(param,'profile'))
			profile = param.profile;
		end
		if (isfield(param,'step'))
			step0 = param.step;
			alpha = 1 / step0;
		end
		if (isfield(param,'tol_off'))
			tol_off = param.tol_off;
		end
		if (isfield(param,'maxtime'))
			maxtime_flag = 1;
		end
		if (isfield(param,'delta'))
			delta = param.delta;
		end
		if (isfield(param,'obj'))
			objflag = 1;
			stopobj = param.obj;
		end
		if (nargin >= 3)
			test_flag = 1;
		end
	end
	if (test_flag == 1)
		if (interchanged)
			Mtest = Mtest';
		end
		[rows_t,cols_t,vals_t] = find(Mtest);
	end

	k = min(m,k);
	maxk = min(m,maxk);

	if (base_iter <= 0)
		pg_iter = maxiter+1;
	end
	%k = min(k,maxNumCompThreads);


	k = min(k,m);
	timelist = [0];
	[rows,cols,vals] = find(M);
	idx0 = find(M);


	Res = sparse(rows,cols,vals,m,n);
	Rbase = zeros(m,0);
	filtereps = filtereps0;
	consecutive_PG = 0;
	truncatedW = zeros(m,0);

	tol = 1e-2;
	tolsvd = 1;
	totaltime = 0;
	kSVDlist = [];
	kMFlist = [];

	oldx_norm2 = 0;
	x_norm2 = 0;
	x0x = 0;

	if (~isfield(param,'W') || ~isfield(param,'H'))
		initobj = norm(M,'fro')^2/2;
		initloss = initobj;
		loss = initloss;
		if (test_flag)
			W = zeros(m,0);
			H = zeros(n,0);
			tempS = speye(size(W, 2), size(H, 2));
			initRMSE = MatCompRMSE(W, H, tempS, rows_t, cols_t, vals_t);
			RMSElist = [initRMSE];
		end
		objlist = [initobj];
		t = tic;
		kSVDlist = [k];
		MMTmap = @(x) step0 * Mt'*(M'*x);
		[W,S] = NystromHMT(MMTmap, m, k); %W: m*k; S: k*k
		H = (Mt*W);%H:n*k, W*S*W' = M*Mt, or W*sqrt(S) * V0' = M; H ~= V0*sqrt(S)
		[U,S,V] = svd(H,0);%U:n*k, S2:k*k, V:k*k;  W * (V0*sqrt(S))' ~= M ==> W * (U*S2*V')' ~= M ==> (W*V) * S2 * U'
		S = diag(S);
		Sidx = find(S <= lambda/alpha);
		if (length(Sidx) > 0)
			Svals = S(Sidx);
			[~,Sidx2] = sort(Svals,'descend');
			Sidx_final = Sidx(Sidx2);
		end

		S = max(S - lambda/alpha, 0);
		S2 = sqrt(S)';
		idx = find(S2 > 0);
		if length(idx) == length(S2)
			%H = (Mt*(W ./ S));
			%W = W .* S;
			H = U .* S2;
			W = W * (V .* S2);
		else
			k = length(idx);
			H = U(:,idx) .* S2(idx);
			truncatedW = W * V(:,Sidx_final);
			W = W * (V(:,idx) .* S2(idx));
		end
		Resvec = partXY(W' ,H', rows, cols, length(rows)) - vals;

		newloss = (Resvec'*Resvec)/2;
		newobj = newloss + lambda / 2 * (norm(H,'fro')^2 + norm(W,'fro')^2);

		objlist = [objlist;newobj];
		totaltime = totaltime + toc(t);
		timelist = [timelist;totaltime];
		Rbase = W;
		tol = (initobj - newobj) / abs(newobj);
		tolsvd = (initobj - newobj);
		proxiters = [2];
		x_norm2 = S'*S;
		consecutive_PG = 1; % The initialization step is essentially one PG step
	else
		if (interchanged)
			W = param.H;
			H = param.W;
		else
			W = param.W;
			H = param.H;
		end
		x_norm2 = sum(sum((W'*W).*(H'*H)));
		Resvec = partXY(W' ,H', rows, cols, length(rows)) - vals;
		initloss = (Resvec'*Resvec)/2;
		initobj = initloss + lambda / 2 * (norm(H,'fro')^2 + norm(W,'fro')^2)
		newobj = initobj;
		newloss = initloss;
		objlist = [newobj];
		timelist = [0];
		if(test_flag)
			RMSElist = [];
		end
		k = size(W,2); %Assume that U is m*k and V is n*k without error tolerance
		kSVDlist = [k];
	end
	if(test_flag)
		tempS = speye(size(W, 2), size(H, 2));
		RMSElist = [RMSElist;MatCompRMSE(W, H, tempS, rows_t, cols_t, vals_t)];
	end
	oldk = 0;
	options.maxit = svditer;
	options.profile = profile;

	%tol = 1e-3;

	inner_iter = base_iter;

	flag_skipBM = (consecutive_PG < pg_iter || base_iter == 0);
	if (~flag_skipBM)
		oldgrad = Resvec;
	else
		oldgrad = -vals;
	end

	flag_subset = 0;
	backtrackinglist = [];
	stepsizelist = [];
	prevk = k;
	flag_add_rank = 0;
%%%
%%%Entering the main loop
%%%
	for it=1:maxiter
		t = tic;
		oldk = k;
		oldkprint = oldk;
		kMFlist = [kMFlist;k];
		timer = tic;
		objchangeBM = 0;
		if (flag_skipBM || base_iter == 0)
			oldkprint = -1;
			obj = newobj;
			loss = newloss;
			pmtime = 0;
			polyiter = 0;
		else
			consecutive_PG = 0;
			if (tol_off == 0)
				cmd = ['-l ', num2str(lambda), ' -n ', num2str(threads), ' -E ', num2str(max(min(tol,tol^(exponent)),1e-20)), ' -t ', num2str(inner_iter), ' -q ' num2str(profile) ' -k ', num2str(min(k,maxk))];
			else
				cmd = ['-l ', num2str(lambda), ' -n ', num2str(threads), ' -E ', num2str(eps^2), ' -t ', num2str(inner_iter), ' -q ' num2str(profile) ' -k ', num2str(min(k,maxk))];
			end

			if (profile == 1)
				fprintf('Enter BM step\n');
			end
			oldgrad = Resvec;

			oldW = W;
			oldH = H;

			[W,H,Resvec,obj,pmftime,polyiter,innerobjlist,innertimelist] = pmf_train_matlab(M,Mt,W,H,cmd);
			oldx_norm2 = x_norm2;
			x0x = sum(sum((oldW'*W).*(oldH'*H)));
			loss = (Resvec'*Resvec)/2;
			x_norm2 = sum(sum((W'*W).*(H'*H)));
			objchangeBM = newobj - obj;
			objlist = [objlist; innerobjlist];
			timelist = [timelist; innertimelist + totaltime];
		end
		subprobtime = toc(timer);

		setSval(Res, Resvec, length(Resvec));

		normx_x02 = x_norm2 + oldx_norm2 - 2 * x0x;
		step = step0;
		alpha = 1 / step;

		if (step ~= 1)
			Amap = @(x) AATmap(W, H, Res, x, step);
			UTA = @(U) (W'*U)'*H' - step * U'*Res;
		else
			Amap = @(x) AATmap(W, H, Res, x);
			UTA = @(U) (W'*U)'*H' - U'*Res;
		end

		timer = tic;
		sig = sqrt(sum(W.^2,1));
		if (flag_add_rank)
			additional_col = rand(m,1)*2-1;
			additional_col = additional_col / (noise_shrink*norm(additional_col));
			if (size(truncatedW,2) > 0)
				additional_col = additional_col + truncatedW(:,1);
				truncatedW = truncatedW(:,2:end);
			end
			additional_col = additional_col - Rbase*(Rbase'*additional_col);
			additional_col = additional_col / norm(additional_col);
			Rbase = [Rbase, additional_col / norm(additional_col)];
		end

		R = filterBase(W, Rbase, sig, filtereps);%Combine W and oldW as possible eigenvectors, but remove redundant ones by doing some Gram-Schmidt like steps

		k = min(size(R,2),m);
		kSVDlist = [kSVDlist;k];

		consecutive_PG = consecutive_PG + 1;
		flag_skipBM = (consecutive_PG < pg_iter || base_iter == 0);

		oldH = H;
		oldW = W;
		oldx_norm2 = x_norm2;

		if (flag_skipBM)
			oldgrad = Resvec;
			if (it > 1)
				if (flag_subset)
					Rbase = U(:,idx);%From fastsvd below
				else
					Rbase = U;
				end
			end
		end

		kinput = k;
		svdtime = 0;
		backtracking = 0;
		svd_total_iters = 0;
		svd_total_time = 0;
		options.tol = tolsvd;
		grad = Resvec;
		while(1)
			if (profile)
				fprintf('\nRunning approximate SVD\n');
			end
			[U,S,V,info0] = fastsvd(Amap, m, kinput, R, UTA, options);
			info0.iter;
			svd_total_iters = svd_total_iters + info0.iter;
			svd_total_time = svd_total_time + info0.ttime;

			R = U;
			svdtime = svdtime + toc(timer);
			Sidx = find(S <= lambda/alpha);
			if (length(Sidx) > 0)
				Svals = S(Sidx);
				[~,Sidx2] = sort(Svals,'descend');
				Sidx_final = Sidx(Sidx2);
				truncatedW = U(:,Sidx_final);
			end
			S = max(S - lambda / alpha, 0);
				

			x_norm2 = S'*S;
			k0 = k;
			if (min(S) <= 0)
				flag_subset = 1;
				idx = find(S > 0);
				S = S(idx);
				D = sqrt(S)';
				k = length(idx);
				if (~flag_skipBM)
					Rbase = U(:,idx);
				end
				if (length(idx) > 0)
					W = U(:,idx) .* D;
					H = V(:,idx) .* D;
				else
					W = zeros(m,0);
					H = zeros(n,0);
				end
			else
				flag_subset = 0;
				D = sqrt(S)';
				if (~flag_skipBM)
					Rbase = U;
				end
				W = U .* D;
				H = V .* D;
			end

			timer = tic;
			x0x = sum(sum((oldW'*W).*(oldH'*H)));
			normx_x02 = x_norm2 + oldx_norm2 - 2 * x0x;

			Resvec = partXY(W' ,H', rows, cols, length(rows),threads) - vals;
			newloss = (Resvec'*Resvec)/2;

			newobj = newloss + sum(S) * lambda;

			if ((newloss < loss + grad'*(Resvec - grad) + normx_x02 * delta/step ))
				break;
			else
				backtracking = backtracking + 1;
				alpha = alpha * eta;
				step = 1 / alpha;
				Amap = @(x) AATmap(oldW, oldH, Res, x, step);
				UTA = @(U) (oldW'*U)'*oldH' - step * U'*Res;
			end

			if (step <= step_lower_bound)
				stop_flag = 1;
				step = 0;
				it = it - 1;
				newobj = obj
				newloss = loss
				fprintf('Stopped: backtracking fails\n');
				break;
			end
			timer = tic;
		end
		if (k == kinput - 1 && oldk == k)
			noise_shrink = noise_shrink * 2;
		else
			noise_shrink = 10;
		end
		backtrackinglist = [backtrackinglist;backtracking];
		stepsizelist = [stepsizelist;step];
		if (stop_flag)
			break;
		end

		if ((kinput <= k+1) && (oldk == k))
			flag_add_rank = 1;
		else
			flag_add_rank = 0;
		end

		tol = abs(obj - newobj) / abs(newobj);
		tolsvd = abs(obj - newobj);
		tolsvd = normx_x02;
		tol = normx_x02 / max(1,abs(newobj));
		objlist = [objlist;newobj];
		proxiters = [proxiters; length(objlist)];

		k = length(S);

		if (k == oldk)
			filtereps = filtereps1;
			if (polyiter < inner_iter - 1)
				tol_off = 1;
			end
		else
			filtereps = filtereps0;
			inner_iter = base_iter;
		end
		totaltime = totaltime + toc(t);
		timelist = [timelist;totaltime];
%Code from below on till end of iter should not be counted for timing
		newRMSE = -1;
		if(test_flag)
			tempS = speye(size(W, 2), size(H, 2));
			newRMSE = MatCompRMSE(W, H, tempS, rows_t, cols_t, vals_t);
			RMSElist = [RMSElist;newRMSE];
		end


		fprintf('iter %3d obj %10.15e elapsed_time %5.3e RankforSVD %3d RankAfterSVD %3d RankInSubprob %3d stepsize %5.3e backtracking %d objchange in proxgrad %5.3e objchange in BM %5.3e RMSE %5.3e\n',it, newobj, totaltime, kinput, k, oldkprint, step, backtracking, obj - newobj, objchangeBM,newRMSE);
		if (profile == 1)
			fprintf('\t\tSubproblem iter %d time %5.3e\t\t SVD iter %d time %5.3e\n',polyiter, subprobtime, svd_total_iters, svd_total_time);
		end

		if (normx_x02 / (abs(newobj) + 1) < eps)
			break;
		end
		if (abs(newobj - obj) / abs(obj) < eps)
			break;
		end
		if (objflag && newobj <= stopobj)
			break;
			info.reachtime = timelist(end);
		end
		if (maxtime_flag && (timelist(end) >= param.maxtime))
			break;
		end
	end
	time = totaltime;
	info.obj = min(newobj,obj);
	info.time = time;
	info.objlist = objlist;
	info.timelist = timelist;
	info.outerobjlist = objlist(proxiters);
	info.outertimelist = timelist(proxiters);
	info.proxiters = proxiters;
	info.kSVDlist = kSVDlist;
	info.kMFlist = kMFlist;
	info.iter = it;
	if (test_flag)
		info.RMSElist = RMSElist;
	end
	info.backtrackinglist = backtrackinglist;
	info.stepsizelist = stepsizelist;

	if (interchanged)
		x.U = H;
		x.V = W;
	else
		x.U = W;
		x.V = H;
	end
end
