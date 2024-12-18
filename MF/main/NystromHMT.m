function [Vhat, Shat] = NystromHMT(Amap, m, k, HMT)
% Conduct the rank-k Nystrom +HMT approximation for A (symmetric, PSD) using a sketch matrix V
% Ahat = (AV) * pinv(V'AV) * (AV)'
%
% Input:
% Amap: either a function handle or a symmetric matrix of size m*m
% k is the target rank of the approximation
% HMT: 0: HMT step off
%      1: HMT step on: replace the sketch matrix V with orth(A V), namely one step of power method
%
% Output:
% Vhat is the approximate eigenvectors
% Shat is the approximate eigenvalues

	V0 = randn(m,k);
	if (nargin < 3)
		k = 5; %Tentative assigment
	end
	if (nargin < 4)
		HMT = 1;
	end
	k = ceil(k);
	if (k > m)
		k = m; %But in this case just do eigendecomposition is faster
	end

	if (strcmp(class(Amap), 'function_handle'))
		A = @(x) Amap(x);
	else
		At = Amap';
		A = @(x) At * x;
	end

	AV0 = A(V0);
	if (HMT > 0)
		[V,~] = qr(AV0, 0); %V = orth( (A) V)
		Y = A(V); % Y = (A) * V
	else
		V = V0;
		Y = AV0;
	end
	[V,~] = qr(AV0, 0); %V = orth( A V)
	Y = A(V); % Y = A * V

	M = V'*Y;

	[U, Sigma] = eig(M); %V'*A*V >= 0 iff Ahat >= 0
	Sigma = diag(real(Sigma));

	[Q,R] = qr(Y, 0);
	RU = R*U;
	M2 = (RU .* (1./Sigma)') * RU'; % Find the eigendecomposition of Ahat = Q * M2 * Q'
	M2 = (M2+M2')/2;
	[U2, S2] = eig(M2); 
	S2 = max(diag(S2), 0);
	idx = find(S2 > 0);
	if (idx == 0)
		Vhat = zeros(m,1);
	else
		Vhat = Q * U2(:,idx);
		Shat = S2;
	end
end
