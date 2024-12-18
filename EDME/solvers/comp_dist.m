function D = comp_dist(A, idxI, idxJ)
n = size(A, 2);
dd = sqrt(sum((A(:,idxI) - A(:,idxJ)).^2));
%m = length(idxI);
%dd = zeros(m,1);
%for kk = 1:m
%    i = idxI(kk);
%    j = idxJ(kk);
%    Aij = A(:,i) - A(:,j);
%    dd(kk) = sqrt(sum(Aij.*Aij));
%end
D = spconvert([[idxI, idxJ, dd];[n, n, 0]]);
end
