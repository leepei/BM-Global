function [v, info] = manopt_edme_new(idxI, idxJ, ww, dd, lambda, n, k, options, v0)
tstart = clock;
maxiter = 1000;
stoptol = 1e-6;
printyes = 2;
m = length(idxI);
threads = maxNumCompThreads;

%% options
if isfield(options, 'maxiter');  maxiter  = options.maxiter;  end
if isfield(options, 'stoptol');  stoptol  = options.stoptol;  end
if isfield(options, 'printyes'); printyes = options.printyes; end
if isfield(options, 'threads'); threads = options.threads; end

%% intial point
if nargin < 9; v0 = randn(k,n); end

%% define problem structure
problem.M = centeredmatrixfactory(k,n);

%% define cost function
problem.cost = @cost;
    function [f, store] = cost(v, store)
        if ~isfield(store, 'ev')
            store.ev = mexlinmapedme(v, idxI, idxJ);
            %store.ev = lin_map_edme(v, idxI, idxJ);
        end
        ev = store.ev;
        f = obj_new(v, lambda, ww, dd, ev);
    end
%% define Euclidean gradient (usual gradient)
problem.egrad = @egrad;
    function [g, store] = egrad(v, store)
        if ~isfield(store, 'ev')
            store.ev = mexlinmapedme(v, idxI, idxJ);
            %store.ev = lin_map_edme(v, idxI, idxJ);
        end
        ev = store.ev;
	
        if ~isfield(store, 'gtmp')
            store.gtmp = 4*ww .* (ev - dd);
        end
        gtmp = store.gtmp;
        g = mexgradedme_manopt(v, idxI, idxJ, gtmp, lambda);
%        gtmp = ww .* (ev - dd);
%		egrad_EDME(gtmp, idxI, idxJ, v, g, threads);
%         g = 2*lambda*v;
%         for kk = 1:m
%             i = idxI(kk);
%             j = idxJ(kk);
%             gk = gtmp(kk);
%             vk = v(:, i) - v(:, j);
%             g(:, i) = g(:, i) + gk*vk;
%             g(:, j) = g(:, j) - gk*vk;
%         end
    end
%% define Euclidean hssian mapping
problem.ehess = @ehess;
    function [h, store] = ehess(v, u, store)
        if ~isfield(store, 'ev')
%            store.ev = mexlinmapedme(v, idxI, idxJ);
            store.ev = lin_map_edme(v, idxI, idxJ);
        end
        ev = store.ev;
        h = 2*lambda*u;
        if ~isfield(store, 'gtmp')
            store.gtmp = 4*ww .* (ev - dd);
        end
        gtmp = store.gtmp;
		ehess_EDME(v,u,idxI, idxJ, gtmp, ww, h, threads);
        
%        gtmp = ww .* (ev - dd);
%        h = mexhessedme_manopt(v, u, idxI, idxJ, gtmp, ww, lambda);
%         h = 2*lambda*u;
%         for kk = 1:m
%             i = idxI(kk);
%             j = idxJ(kk);
%             vij = v(:, i) - v(:, j);
%             uij = u(:, i) - u(:, j);
%             hij = 8 * ww(kk) * sum(vij .* uij);
%             gij = 4 * gtmp(kk);
%             h(:,i) = h(:,i) + hij*vij + gij*uij;
%             h(:,j) = h(:,j) - hij*vij - gij*uij;
%         end
    end
%% numerically chceck whether gradient or hessian is correct
% checkgradient(problem);
% checkhessian(problem);
%% options for manopt
opts.verbosity = printyes;
opts.maxiter = maxiter;
opts.tolgradnorm = stoptol;
%% solve by trust-region method, most reliable
v = trustregions(problem, v0, opts);
%% record info
info.cputime = etime(clock, tstart);
% info.evt = lin_map_edme(v, idxI, idxJ);
end
