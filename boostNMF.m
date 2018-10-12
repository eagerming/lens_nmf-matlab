function [W, H, iter, grad] = boostNMF(A, dim, params)
%%NMFSH_COMB_RANK2 - A fast algorithm for rank-2 nonnegative matrix factorization
% [W, H, iter, grad] = nmfsh_comb_rank2(A, Winit, Hinit, params)
%
% Input parameters
% A: m*n data matrix
% Winit: m*2 matrix for initialization of W
% Hinit: 2*n matrix for initialization of H
% params (optional)
% params.vec_norm (default=2): indicates which norm to use for the normalization of W or H,
%                              e.g. vec_norm=2 means Euclidean norm; vec_norm=0 means no normalization.
% params.normW (default=true): true if normalizing columns of W; false if normalizing rows of H
% params.tol (default=1e-4): tolerance parameter for stopping criterion
% params.maxiter (default=10000): maximum number of iteration times
%
% Output parameters
% W, H: result of rank-2 NMF
% iter: number of ANLS iterations actually used
% grad: relative norm of projected gradient, reflecting the stationarity of the solution
%
% Da Kuang, Haesun Park
% Feb 2013


%% Parameter Checking and Setting ======================================
if ~exist('params', 'var')
    params = [];
end

if isfield(params, 'vec_norm')
    vec_norm = params.vec_norm;
else
    vec_norm = 2.0;
end
if isfield(params, 'isnormW')
    isnormW = params.isnormW;
else
    isnormW = true;
end
if isfield(params, 'tol')
    tol = params.tol;
else
    tol = 1e-4;
end
if isfield(params, 'maxiter')
    maxiter = params.maxiter;
else
    maxiter = 10000;
end
if isfield(params, 'beta')
    beta = params.beta;
else
    beta = 0;
end

%===================================================
%% 
[W,H] = initialization(A, dim);
left = H * H';
right = A * H';

if rank(left) < dim
    [W,H] = initialization(A, dim);
    left = H * H';
    right = A * H';
end
    
if dim >= 3
    param_sparse = [-1 beta];
    [W, H, iter] = nmfsh_comb(A, dim, param_sparse);
    grad = 0;
    return;
end


for iter = 1 : maxiter
    if rank(left) < dim
		fprintf('The matrix H is singular\n');
    end
    
    if dim == 1        
        W = right ./ (left);
    elseif dim == 2
        W = sparse_rank2_LS(left, right);
    end
    
    W(W<0) = 0;
    
    norms_W = sqrt(sum(W.^2));
	if min(norms_W) < eps
		error('Error: Some column of W is essentially zero');
	end
    
% 	W = W ./ norms_W;
    W = bsxfun(@times, W, 1./norms_W);
    
	left = W' * W;
	right = A' * W;
    
    right = right - beta;
    right(right<0) = 0;
    
    if sum(right(:)) == 0
        error('The sparse coefficient beta is too big to obtain a positive result\n');
    end
    
	if rank(left) < dim
		fprintf('The matrix W is singular\n');
    end
    
    if dim == 1
        H = (right ./ left)';
%         H = ((right - beta) ./ left)';  %  (A' - H'W') - beta|H|_1
%         H = (right ./ (beta + left))';  %  (A' - H'W') - beta|H|_1^2
    elseif dim == 2
        H = sparse_rank2_LS(left, right)';
    end
    
    H(H<0) = 0;
    gradH = left * H - right';
    
    
	left = H * H';
	right = A * H';
	gradW = W * left - right;
	if iter == 1
	        initgrad = sqrt(norm(gradW(gradW<=0|W>0))^2 + norm(gradH(gradH<=0|H>0))^2);
		continue;
	else
        	projnorm = sqrt(norm(gradW(gradW<=0|W>0))^2 + norm(gradH(gradH<=0|H>0))^2);
	end
	if projnorm < tol * initgrad || abs(projnorm) < eps * 100
		break;
	end
end

grad = projnorm / initgrad;

if vec_norm ~= 0
	if isnormW
        	norms = sum(W.^vec_norm) .^ (1/vec_norm);
	        W = bsxfun(@rdivide, W, norms);
        	H = bsxfun(@times, H, norms');
    else    
%         if norm_vertical    % vertically normalize every column of coefficient matrix H.
%         else                % horizentally normalize every row of H.
        	norms = sum(H.^vec_norm, 2) .^ (1/vec_norm);
	        W = bsxfun(@times, W, norms');
        	H = bsxfun(@rdivide, H, norms);
%         end
	end
end

end

%% Initialization ==============================================
function [W,H] = initialization(A, dim)

[m, n] = size(A);

Winit =  rand(m,dim);
Winit(Winit<0) = 0;
Hinit = rand(dim,n);
Hinit(Hinit<0) = 0;

W = Winit;
H = Hinit;
end
