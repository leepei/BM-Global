function relkkt = rel_kkt_full(blk, At, C, X)
x = svec(blk, X);
c = svec(blk, C);
Rp = (x' * At)';
gradf = At*Rp + c;
gradfmat = smat(blk, x-gradf);
[V, D] = eig(gradfmat);
d = diag(D);
idx = find(d > 0);
V = V(:, idx);
d = d(idx);
gradfmatp = V * (d .* V');
gradfp = svec(blk, gradfmatp);
relkkt = norm(x - gradfp) / (1 + norm(x) + norm(gradf));
end