function ex = emap_edme(x)
n = length(x);
en = ones(n,1);
ex = x - en*(en'*x/n);
end