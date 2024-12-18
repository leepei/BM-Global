function f = obj_new(vt, lambda, ww, dd, ev)
ftmp = ev - dd;
f = lambda*norm(vt,'fro')^2 + sum(ww .* (ftmp .* ftmp));
end