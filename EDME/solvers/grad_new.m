function g = grad_new(xi, idxI, idxJ, gtmp, lambda)
g = lambda*xi;
m = length(idxI);
for k = 1:m
    i = idxI(k);
    j = idxJ(k);
    xitmp = xi(i) - xi(j);
    gtmpk = 2*gtmp(k);
    g(i) = g(i) + gtmpk*xitmp;
    g(j) = g(j) - gtmpk*xitmp;
end
end