function [v, info] = apg_edme_new(idxI, idxJ, ww, dd, lambda, n, k, options, v0)
tstart = clock;
maxiter = 10000;
stoptol = 1e-6;
printyes = 1;
useacc = 1;
lipconst = 1e2;
alpfix = 0;

%% 
if isfield(options, 'maxiter');  maxiter  = options.maxiter;  end
if isfield(options, 'stoptol');  stoptol  = options.stoptol;  end
if isfield(options, 'printyes'); printyes = options.printyes; end
if isfield(options, 'useacc');   useacc   = options.useacc;   end
if isfield(options, 'lipconst'); lipconst = options.lipconst; end
if isfield(options, 'alpfix');   alpfix   = options.alpfix;   end

%% 
if nargin < 9; v0 = 0*emap_edme(rand(n,k)); end %v0 = (eye(n) - ones(n)/n)*randn(n,k);
v = v0;
vt = v';
% evt = lin_map_edme(vt, idxI, idxJ);
evt = mexlinmapedme(vt,idxI,idxJ);
objf = obj_new(vt, lambda, ww, dd, evt);

lip = 4 * sum(ww);
alpha = min(1e-2, lipconst/lip);

%% 
breakyes = 0;
msg = [];
if useacc == 1; tnew = 1.0; end
eut = evt;
en = ones(n,1);
projopts = 1;
for iter = 1:maxiter
    %%
    objold = objf;
    evtold = evt;
    %%
    if alpfix == 1
        %% fixed step size
        gtmp = ww .* (eut - dd);
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
        optsproj.smtol = 1/n;
        %if iter > 1; optsproj.v0 = V; end
        [v, k, V, d] = proj_sdp_edme_new(gmap, n, k, optsproj);
        vt = v';
        %evt = lin_map_edme(vt, idxI, idxJ);
        evt = mexlinmapedme(vt, idxI, idxJ);
        objf = obj_new(vt, lambda, ww, dd, evt);
    end
    relobj = abs(objf-objold)/(1+abs(objold));
    
    %% 
    info.rrank(iter) = k;
    info.objf(iter) = objf;
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
    if printyes == 1 && ((iter <= 1000 && rem(iter, 100) == 1) || (iter > 1000 && rem(iter, 500) == 1 ) || breakyes > 0)
        fprintf("\n iter = %5d| obj = %- 9.8e| relobj = %2.1e| rrank = %5d| time = %5.1f| alp = %2.1e", ...
            iter, objf, relobj, k, etime(clock, tstart), alpha);
    end
    %% 
    if (breakyes > 0)
        %fprintf("\n %s \n", msg);
        break;
    end
    %% 
    if useacc == 1
        told = tnew;
        tnew = (1 + sqrt(1 + 4*told^2)) / 2;
        %% restart
        if (iter > 1000 && rem(iter, 200) == 0)
            told = 1.0;
            tnew = 1.0;
            %fprintf("[R]");
        end        
        beta = (told - 1) / tnew;
        eut = (1 + beta) * evt - beta * evtold;
    else
        eut = evt;
    end
end

%% 
info.iter = iter;
info.cputime = etime(clock, tstart);
info.msg = msg;
info.vt = vt;
info.V = V;
info.d = d;
info.evt = evt;
end