function y = lin_map_edme(vt, idxI, idxJ)
y = sum((vt(:,idxI) - vt(:,idxJ)).^2)';
%m = length(idxI);
%y = zeros(m, 1);
%for kk = 1:m
%    i = idxI(kk);
%    j = idxJ(kk);
%    vi = vt(:, i);
%    vj = vt(:, j);
%    vij = vi - vj;
%    y(kk) = sum(vij .* vij);
%end
end
