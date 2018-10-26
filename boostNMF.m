function [W_return, H_return, isSuccess, iter, cost] = boostNMF(R, dim, params)
%%NMFSH_COMB_RANK2
% Input parameters
% A: m*n data matrix
% dim: dimension of the factors.
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

% if isfield(params, 'vec_norm')
%     vec_norm = params.vec_norm;
% else
%     vec_norm = 2.0;
% end
% if isfield(params, 'isnormW')
%     isnormW = params.isnormW;
% else
%     isnormW = true;
% end
if isfield(params, 'tol')
    tol = params.tol;
else
    tol = 1e-4;
end
if isfield(params, 'max_iter')
    max_iter = params.max_iter;
else
    max_iter = 100;
end
if isfield(params, 'learning_rate')
    learning_rate = params.learning_rate;
else
    learning_rate = 0.01;
end
if isfield(params, 'lambda')
    lambda = params.lambda;
else
    lambda = 0;
end
if isfield(params, 'lambda_social')
    lambda_social = params.lambda_social;
else
    lambda_social = 0;
end
if isfield(params, 'lambda_item')
    lambda_item = params.lambda_item;
else
    lambda_item = 0;
end

if lambda_social > 0 && isfield(params, 'social_matrix')
	social_matrix = params.social_matrix;
else
    social_matrix = sparse(eye(size(R,2)));
end
if lambda_item > 0 && isfield(params, 'item_matrix')
	item_matrix = params.item_matrix;
else
    item_matrix = sparse(eye(size(R,1)));
end




if ~isfield(params, 'is_zero_mask_of_missing') 
    is_zero_mask_of_missing = false;
else
    is_zero_mask_of_missing = params.is_zero_mask_of_missing;
end

if is_zero_mask_of_missing
    if ~isfield(params, 'mask') 
        error('Mask abscent!');
    else
        mask = params.mask;
    end
end


%===================================================
%% 
col_ind = sum(abs(R),1) > 0;
row_ind = sum(abs(R),2) > 0;
R_original = R;
W_return = zeros(size(R_original,1), dim);
H_return = zeros(dim, size(R_original,2));


R = R(row_ind, col_ind);
item_matrix = item_matrix(row_ind, row_ind);
social_matrix = social_matrix(col_ind, col_ind);
mask = mask(row_ind, col_ind);


[W,H] = initialization(R, dim);


num_row = size(R,1);
num_col = size(R,2);
cost = -1;

flr = 1e-4;
isSuccess = true;

for iter = 1 : max_iter
    
    social_W = (W' * item_matrix)';
    for i = 1:num_row
        if is_zero_mask_of_missing
            indicate_vec = mask(i,:);
        else
            indicate_vec = true(1,num_col);
        end
        left = H(:,indicate_vec) * H(:,indicate_vec)';
        right = R(i,:) * H';
        social_Wi = social_W(i,:);
        numerator_Wi = right + lambda_item * social_Wi;
        denominator_Wi = left + lambda + lambda_item;
        denominator_Wi(denominator_Wi==0) = flr;
        
        if dim == 1
            W_i = numerator_Wi ./ denominator_Wi;
        elseif dim == 2
            if denominator_Wi(1,1)==0 || denominator_Wi(2,2)==0 
                isSuccess = false;
                disp('[Warning] The Matrix H is singular, reduce the dimension by 1');
                return;
            end
            W_i = sparse_rank2_LS(denominator_Wi, numerator_Wi);
        else
            dpw = W(i,:) * denominator_Wi;
            if learning_rate ~= 0
                W_i = W(i,:) + params.learning_rate * (numerator_Wi - dpw);
            else
                dpw(dpw == 0) = flr;
                W_i = W(i,:) .* numerator_Wi ./ (dpw);
            end

        end
        W(i,:) = W_i;
    end
    
    norms_W = sqrt(sum(W.^2));
    if min(norms_W) < eps
		error('Error: Some column of W is essentially zero');
%         is_success = false;
%         return;
    end
    W = bsxfun(@times, W, 1./norms_W);
    H  = bsxfun(@times, H, norms_W');
    
%     if params.is_zero_mask_of_missing
%         L = mask_result(R, W, H);
%     else
%         L = W * H;
%     end
%     costa = L - R;
%     costa = sum(sum(costa.^2))
%     
%     
    social_H = (H * social_matrix);
    for j = 1:num_col
        if is_zero_mask_of_missing
            indicate_vec = mask(:,j);
        else
            indicate_vec = true(num_row,1);
        end
        
        left = W(indicate_vec, :)' * W(indicate_vec, :);
        right = R(:, j)' * W;
        social_Hj = social_H(:,j);
        numerator_Hj = right + lambda_social * social_Hj';
        denominator_Hj = left + lambda + lambda_social;
        denominator_Hj(denominator_Hj==0) = flr;
        
        if dim == 1
            H_j = (numerator_Hj ./ denominator_Hj)';
        elseif dim == 2
            if denominator_Hj(1,1)==0 || denominator_Hj(2,2)==0 
                isSuccess = false;
                disp('[Warning] The Matrix is singular, reduce the dimension by 1');
                return;
            end
            H_j = sparse_rank2_LS(denominator_Hj, numerator_Hj)';
        else
            dph = denominator_Hj * H(:,j);
            if learning_rate ~= 0
                H_j = H(:, j) + params.learning_rate * (numerator_Hj' - dph);
            else
                dph(dph == 0) = flr;
                H_j = H(:, j) .* numerator_Hj' ./ dph;
            end
        end
        H(:, j) = H_j;
    end

    
    if params.is_zero_mask_of_missing
        L = mask_result(R, W, H);
    else
        L = W * H;
    end
    cost = L - R;
    cost = sqrt(sum(sum(cost.^2)));
    
    if iter > 1 && tol > 0
        e = abs(cost - last_cost) / last_cost;
        if cost > last_cost
            if dim == 2 || dim == 1
                break;
            end
            if params.learning_rate ~= 0
                if cost/last_cost > 10
                    error('Learning rate is too large!');
                else
                    params.learning_rate = params.learning_rate /5;
                    disp('half the learning rate');
                end
            end
        end
        if (e < tol)
%             disp('Convergence reached, aborting iteration') 
%             objective.div = objective.div(1:it); 
%             objective.cost = objective.cost(1:it);
            break
        end
    end
    last_cost = cost;
end

W_return(row_ind,:) = W;
H_return(:, col_ind) = H;


% if vec_norm ~= 0
% 	if isnormW
%         	norms = sum(W.^vec_norm) .^ (1/vec_norm);
% 	        W = bsxfun(@rdivide, W, norms);
%         	H = bsxfun(@times, H, norms');
%     else    
%         	norms = sum(H.^vec_norm, 2) .^ (1/vec_norm);
% 	        W = bsxfun(@times, W, norms');
%         	H = bsxfun(@rdivide, H, norms);
% 	end
% end

end

%% Initialization ==============================================
function [W,H] = initialization(A, dim)

[m, n] = size(A);

Winit =  rand(m,dim);
% Winit(Winit<0) = 0;
Hinit = rand(dim,n);
% Hinit(Hinit<0) = 0;

W = Winit;
H = Hinit;
end
