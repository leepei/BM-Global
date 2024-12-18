function D = sim2dis(K)
n = size(K,1);
D = zeros(n,n);
for i = 1:n
    for j = 1:n
        D(i,j) = sqrt(K(i,i) + K(j,j) - K(i,j) - K(j,i));
    end
end
D = 0.5*(D+D');
end