function [AATx] = AATmap(W,H,S,x,alpha)
	if (nargin < 5 || alpha == 1)
		y = H*(W'*x)-(x'*S)';
		AATx = W*(H'*y) - (y'*S')';
	else
		y = H*(W'*x)-alpha * (x'*S)';
		AATx = W*(H'*y) - alpha * (y'*S')';
	end
end
