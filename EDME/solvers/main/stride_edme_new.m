function [v, info] = stride_edme_new(idxI, idxJ, ww, dd, lambda, n, k, options, v0)
tstart = clock;
maxiter = 10000;
stoptol = 1e-6;
printyes = 1;
useacc = 1;
lipconst = 1e2;
alpfix = 0;
maxiterAPG = 500;
print_iter = 20;
maxitermanopt = 10;
printmanopt = 0;

%% options
if isfield(options, 'maxiter');  maxiter  = options.maxiter;  end
if isfield(options, 'stoptol');  stoptol  = options.stoptol;  end
if isfield(options, 'printyes'); printyes = options.printyes; end
if isfield(options, 'useacc');   useacc   = options.useacc;   end
if isfield(options, 'lipconst'); lipconst = options.lipconst; end
if isfield(options, 'alpfix');   alpfix   = options.alpfix;   end
if isfield(options, 'maxiterAPG');    maxiterAPG    = options.maxiterAPG;    end
if isfield(options, 'run_bm_iter');   print_iter    = options.run_bm_iter;   end
if isfield(options, 'maxitermanopt'); maxitermanopt = options.maxitermanopt; end
if isfield(options, 'printmanopt');   printmanopt   = options.printmanopt;   end

%% APG for warmstarting
if nargin < 9
    fprintf("\n--------------------------------------------------------------");
    fprintf("--------------------------------------------------------------");
    fprintf("\n APG for warmstarting");
    optionsAPG.maxiter = maxiterAPG;
    optionsAPG.stoptol = stoptol;
    optionsAPG.useacc = useacc;
    optionsAPG.alpfix = alpfix;
    optionsAPG.lipconst = lipconst;
    [v, info0] = apg_edme_new(idxI,idxJ,ww,dd,lambda,n,k,optionsAPG);
    vt = info0.vt;
    evt = info0.evt;
    k = info0.rrank(end);
    objf = info0.objf(end);
    info.info0 = info0;
    info.iter0 = info0.iter;
    info.time0 = info0.cputime;
    fprintf("\n--------------------------------------------------------------");
    fprintf("--------------------------------------------------------------");
else
    v = v0;
    vt = v';
    %evt = lin_map_edme(vt, idxI, idxJ);
    evt = mexlinmapedme(vt, idxI, idxJ);
    objf = obj_new(vt, lambda, ww, dd, evt);
end

lip = 4 * sum(ww);
alpha = lipconst/lip;

%% 
breakyes = 0;
msg = [];
total_iter_manopt = 0;
en = ones(n,1);
projopts = 1;
for iter = 1:maxiter
    %%
    objold = objf;
    %% manopt
    run_manopt = 0;
    if true || rem(iter, 20) == 1; run_manopt = 1; end
    if run_manopt == 1
        if printmanopt 
            fprintf("\n--------------------------------------------------------------");
            fprintf("--------------------------------------------------------------\n");
        end
        optionsMANOPT.maxiter = maxitermanopt;
        optionsMANOPT.stoptol = 1e-6;
        optionsMANOPT.printyes = printmanopt;
        [vt, info_manopt] = manopt_edme_new(idxI, idxJ, ww, dd, lambda, n, k, optionsMANOPT, vt);
        v = vt';
        evt = mexlinmapedme(vt, idxI, idxJ);
        total_iter_manopt = total_iter_manopt + 1;
        info.time_manopt(total_iter_manopt) = info_manopt.cputime;
        if printmanopt 
            fprintf("\n--------------------------------------------------------------");
            fprintf("--------------------------------------------------------------\n");
        end
    end
    %%
    gtmp = ww .* (evt - dd);
    if projopts == 1
        if sign(lambda) < 0
            gmap = @(xi) v*(vt*xi) - alpha*mexgradedme(xi,idxI,idxJ,gtmp,lambda) + (alpha*lambda*sum(xi))*en;
        else
            gmap = @(xi) v*(vt*xi) - alpha*mexgradedme(xi,idxI,idxJ,gtmp,lambda);
        end
    else
        gmaptmp = @(xi) v*(vt*xi) - alpha*mexgradedme(xi,idxI,idxJ,gtmp,lambda);
        gmap = @(x) emap_edme(gmaptmp(emap_edme(x)));
    end
    [v, k, V, d] = proj_sdp_edme_new(gmap, n, k);
    vt = v';
    evt = mexlinmapedme(vt, idxI, idxJ);
    objf = obj_new(vt, lambda, ww, dd, evt);
    relobj = abs(objf-objold)/(1+abs(objold));
    %relkkt = 0; %kkt_edme_new(v, vt, lambda, n, k, idxI, idxJ, ww, dd, evt);
    %% 
    info.rrank(iter) = k;
    info.objf(iter) = objf;
    %info.relobj(iter) = relobj;
    %%
    if relobj < stoptol
        breakyes = 1;
        msg = 'converged';
    end
    if iter == maxiter
        breakyes = 2;
        msg = 'maxiter reached';
    end
    %% 
    if printyes == 1 %&& (rem(iter, print_iter) == 1 || breakyes > 0)
        fprintf("\n iter = %5d| obj = %- 9.8e| relobj = %2.1e | rrank = %4d| time = %5.1f| alp = %2.1e", ...
            iter, objf, relobj, k, etime(clock, tstart), alpha);
    end
    %% 
    if (breakyes > 0)
        %fprintf("\n %s \n", msg);
        fprintf("\n");
        break;
    end
end

%% 
info.iter = iter;
info.cputime = etime(clock, tstart);
info.msg = msg;
info.vt = vt;
info.V = V;
info.d = d;
end
