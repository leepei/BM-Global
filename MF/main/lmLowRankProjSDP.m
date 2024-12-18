%This function is modified from the LmSVD code downloaded from https://www.mathworks.com/matlabcentral/fileexchange/46875-lmsvd-m
%This function implements the LmSVD algorithm proposed by
%Xin Liu, Zaiwen Wen, and Yin Zhang. Limited memory block krylov subspace
%optimization for computing dominant singular value decompositions. SIAM
%Journal on Scientific Computing, 35(3):A1641–A1668, 2013.
%Major changes:
%1. Use lmsvd when the memory is not filled yet: we usually have very few iterations
%2. When break, use the latest Xhat but not X
%3. Cut tolerance set to a variable dtol and relax its value

function [X, dX, info] = lmLowRankProjSDP(Amap, r,n, X0, options)
%% A: n by n symmetric matrix
%% r: rank to project
%% X: eigen-vectors (orthonormal)
%% dX: leading eigenvectors (less than or equal to r)

tstart = clock;
tol = 1e-2;
maxit = 5;
tau = 10;
printyes = 1;
dtol = 1e-6;

if (r <= n*0.02)
    memo = 5;
elseif (r <= n * 0.03)
    memo = 4;
else
    memo = 3;
end


if (nargin == 5)
	if (isfield(options, 'tol'))
		tol = options.tol;
	end
	if (isfield(options,'dtol'))
		dtol = options.dtol;
	end
	if (isfield(options, 'maxit'))
		maxit = options.maxit;
	end
	if (isfield(options, 'memo'))
		memo = options.memo;
	end
	if (isfield(options, 'gvk'))
		tau = options.gvk;
	end
	if (isfield(options,'profile'))
		printyes = options.profile;
	end
end
%dtol = min(dtol,tol);
%% working size
k = min([2 * r, r + tau, n]);
if (k < r)
    error('Working size too small');
end


%% initial guess
if (nargin < 4 || isempty(X0))
    X = randn(n, k);
else
	k0 = size(X0,2);
	if (k0 >= k)
		X = X0(:,1:k);
	else
		X = [X0 randn(n,k-k0)];
	end
end
%[Xm, AXm, X, AX] = initXm(A, X, n, k, memo);
Lm = 0;
%memo
%k
AX = Amap(X);
Xm = X;
AXm = AX;


%% set tolerance
rtol = tol;
%kkttol = rtol;

%%
%% main loop
%% 
breakyes = 0;
dQ = zeros(r, 1);
for iter = 1:maxit
    %% compute P = (I - X_iX_i^T)[X_i-1, ..., X_i-p]
    idxHist = k+1:k+Lm;
    XTXm = X' * Xm(:, idxHist);
	AP = AXm(:, idxHist) - AX * (XTXm);
    P = Xm(:, idxHist) - X*XTXm;
    PTP = P' * P;
    %% stablization, remove small columns in P
    if (Lm > k)
        diagPTP = diag(PTP);
        idxCut = find(diagPTP > dtol);
        L = length(idxCut);
        if (L < 0.95*Lm)
            Lm = L;
            P = P(:, idxCut);
            PTP = PTP(idxCut, idxCut);
            AP = AP(:, idxCut);
        end
    end
    %% eigenvalue decomposition of PTP
    [Up, Dp] = eig(PTP);
    dp = diag(Dp);
    [dp, idxp] = sort(dp, 'descend');
    %% remove small eigenvalues and eigenvectors
    idxlargep = find(dp > dtol);
    if (isempty(idxlargep) && iter > 1)
        %Lm = 0;
		if (printyes)
			fprintf('\n Lm = 0, break');
		end
        break;
    else
        L = length(idxlargep);
    end
    dp = dp(idxlargep);
    Up = Up(:, idxp(idxlargep));
    %% compute Q
	UD = (Up ./sqrt(dp)');
    PUD = P * UD;
    Q = [X, PUD];
	APUD = AP * (UD);
%    APUD = A*PUD;
    AQ = [AX, APUD];
    QTAQ = Q' * AQ;
    QTAQ = (QTAQ + QTAQ')/2;
    %% solve the subspace optimization problem via eigenvalue decomposition of Q
    if (issparse(QTAQ))
        QTAQ = full(QTAQ);
    end
    [Uq, Dq] = eig(QTAQ);
    dq = diag(Dq);
    [dq, idxq] = sort(dq, 'descend');
    Uq = Uq(:, idxq(1:k));
    Xhat = Q * Uq;
    AXhat = AQ * Uq;
    dQ0 = dQ;
    dQ = dq(1:r);
    idxposq = find(dQ > dtol);
    chg_dq = norm(dQ0(idxposq) - dQ(idxposq))/norm(dQ(idxposq));
    %% check for termination (two-level criterion)
    if (chg_dq < rtol && iter > 1)
		X = Xhat;
		AX = AXhat;
        breakyes = 1;
%         kkt = AXhat - (Xhat*Xhat')*AXhat;
%         kkt = kkt(:, 1:r);
%         errkkt = norm(kkt)/normA;
%         if (errkkt < kkttol)
%             break;
%         end 
    end
    %% print iter
    if (printyes)
        fprintf('\n iter = %4d, Lm = %4d, chg_dq = %2.1e, time = %4.3f', ...
            iter, Lm, chg_dq, etime(clock, tstart));
    end
    if (breakyes)
        if (printyes); fprintf('\n converged!'); end
        break; 
    end
    %% next iterate
    [X, ~] = qr(AXhat, 0);
    AX = Amap(X);
	if (memo + 1 < iter)
		Lm = max(0, round(L/k)) * k;
	else
		Lm = k * iter;
	end
    if (Lm == 0)
        Lm = k;
    end
	Xm(:, k+1:k+Lm) = Xm(:, 1:Lm);
	AXm(:, k+1:k+Lm) = AXm(:, 1:Lm);
    Xm(:, 1:k) = Xhat;
    AXm(:, 1:k) = AXhat;
end
XTAX = X' * AX;
dX = diag(XTAX);
[dX, idx] = sort(dX, 'descend');
X = X(:, idx);
idxpos = find(dX > dtol);
rr = min(r, length(idxpos));
dX = dX(1:rr);
X = X(:, 1:rr);
info.iter = iter;
info.ttime = etime(clock, tstart);
if (printyes); fprintf('\n'); end
end

function [Xm, AXm, X, AX] = initXm(Amap, X, n, k, memo)
Xm = zeros(n, (memo + 1) * k);
AXm = zeros(n, (memo + 1) * k);
[X, ~] = qr(X, 0); % economy qr
AX = Amap(X);
Xm(:, end-k+1:end) = X;
AXm(:, end-k+1:end) = AX;
for iter = 1:memo
    [X, ~] = qr(AX, 0);
    AX = Amap(X);
    Xm(:, end - (iter+1)*k + 1 : end - iter*k) = X;
    AXm(:, end - (iter+1)*k + 1 : end - iter*k) = AX;
end
end
