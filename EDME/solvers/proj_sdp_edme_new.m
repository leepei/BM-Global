function [v, k, V, d] = proj_sdp_edme_new(gmap, n, k, options)
if nargin < 4
    options = [];
end
if ~isfield(options, 'smtol'); options.smtol = 1e-6; end
if ~isfield(options, 'v0'); options.v0 = []; end

projopts = 1;
if projopts == 0
    opts.profile = 0;
    opts.maxit = 20;
    opts.tol = 1e-2;
    opts.memo = 5;
    [V, d, ~] = lmLowRankProjSDP2(gmap, k, n, options.v0, opts);
    k = length(d);
    v = V*spdiags(sqrt(d), 0, k, k);
elseif projopts == 1
    opts.issym = 1;
    opts.maxit = 20;
    opts.tol = 1e-12;
    [V, D] = eigs(gmap, n, k, 'LR', opts);
    d = diag(D);
    idxp = find(d > options.smtol & ~isnan(d));
    k = length(idxp);
    V = V(:,idxp);
    d = d(idxp);
    v = V*spdiags(sqrt(d), 0, k, k);
end

end