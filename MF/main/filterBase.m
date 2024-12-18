function [ R ] = filterBase( V1, V0, sig, tol )
%Combine V1, V0, and truncatedV to get an orthonormal basis from them
%Assuming that V1 is already almost orthogonal
%
if (length(V0) == 0)
	[R,~] = qr(V1,0);
elseif (length(V1) == 0)
	[R,~] = qr(V0,0);
else
	R = V0 - V1*((V1'*V0)./(sig.^2)');
	R = sum(R.^2, 1);
	R = (R > tol);
	[R,~] = qr([V1, V0(:, R)],0);
end

end
