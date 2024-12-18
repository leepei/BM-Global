function [blk, At, b, C, dd, ww] = genblk_new(idxI, idxJ, W, D)
n = size(W,1);
blk{1,1} = 's';
blk{1,2} = n;

m = length(idxI);

dd = zeros(m,1);
ww = zeros(m,1);
b  = zeros(m,1);
At = sparse(n*(n+1)/2,m);
C = speye(n,n);

%%
for k = 1:m
    %%
    i = idxI(k);
    j = idxJ(k);
    %%
    dd(k) = D(i,j);
    ww(k) = W(i,j);
    %%
    Itmp = [i; j; i; j; n];
    Jtmp = [i; j; j; i; n];
    Vtmp = [1; 1; -1; -1; 0];
    Eij = spconvert([Itmp, Jtmp, Vtmp])*sqrt(2*ww(k));
    b(k) = dd(k)*sqrt(2*ww(k));
    At(:,k) = svec(blk, Eij, 1);
end

%% Diagonal part
% for k = 1:n
%     dd(k+m) = D(k, k);
%     ww(k+m) = W(k, k);
%     Ek = sparse(n,n);
%     Ek(k, k) = 1;
%     At(:,k+m) = svec(blk, Ek, 1);
% end
end