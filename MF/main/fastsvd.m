function [U,S,V,info] = fastsvd(Amap, n, k, W, UTA, optionsin)
	options.maxit = 5;
	options.profile = optionsin.profile;
	if (isfield(optionsin,'tol'))
		options.tol = optionsin.tol;
	end
	if (isfield(optionsin,'maxit'))
		options.maxit = optionsin.maxit;
	end

	[U, S0, info] = lmLowRankProjSDP(Amap, k, n, W, options);
	S1 = sqrt(S0);
	V0 = UTA(U);
	V = (V0 ./ S1)';
	S = S1;
end
