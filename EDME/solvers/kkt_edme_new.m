function relkkt = kkt_edme_new(v, vt, lambda, n, k, idxI, idxJ, ww, dd, evt)
gmap = @(x) v*(vt*x) - grad_new(x, lambda, idxI, idxJ, ww, dd, evt);
[vkkt, ~, ~, ~] = proj_sdp_edme_new(gmap, n, k);
relkkt = sqrt(norm(vt*v,'fro')^2+norm(vkkt'*vkkt,'fro')^2-2*norm(vt*vkkt, 'fro')^2)/(1+norm(vt,'fro')^2);
end