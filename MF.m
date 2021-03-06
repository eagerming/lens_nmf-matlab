function [W_return, H_return, iter, cost] = MF(R, dim, params)
% SPARSE_NMF Sparse NMF with beta-divergence reconstruction error,
% L1 sparsity constraint, optimization in normalized basis vector space.
%
% [w, h, objective] = sparse_nmf(v, params)
%
% Inputs:
% v:  matrix to be factorized
% params: optional parameters
%       beta:     beta-divergence parameter (default: 1, i.e., KL-divergence)
%       cf: %     cost function type (default: 'kl'; overrides beta setting)
%               'is': Itakura-Saito divergence
%               'kl': Kullback-Leibler divergence
%               'ed': Euclidean distance
%
%
%       sparsity: weight for the L1 sparsity penalty (default: 0)
%       max_iter: maximum number of iterations (default: 100)
%       conv_eps: threshold for early stopping (default: 0,
%                                             i.e., no early stopping)
%       display:  display evolution of objective function (default: 0)
%       random_seed: set the random seed to the given value
%                    (default: 1; if equal to 0, seed is not set)
%
%       init_w: % initial setting for W (default: random;
%                        either init_w or r have to be set)
%       r: % Number of basis functions (default: based on init_w's size;
%                        either init_w or r have to be set)
%       init_h: % initial setting for H (default: random)
%       w_update_ind: set of dimensions to be updated (default: all)
%       h_update_ind: set of dimensions to be updated (default: all)
%
% Outputs:
% w: matrix of basis functions
% h: matrix of activations
% objective: objective function values throughout the iterations
%
%
%
% References:
% J. Eggert and E. Korner, "Sparse coding and NMF," 2004
% P. D. O'Grady and B. A. Pearlmutter, "Discovering Speech Phones
%   Using Convolutive Non-negative Matrix Factorisation
%   with a Sparseness Constraint," 2008
% J. Le Roux, J. R. Hershey, F. Weninger, "Sparse NMF - half-baked or well
%   done?," 2015
%
% This implementation follows the derivations in:
% J. Le Roux, J. R. Hershey, F. Weninger,
% "Sparse NMF - half-baked or well done?,"
% MERL Technical Report, TR2015-023, March 2015
%
% If you use this code, please cite:
% J. Le Roux, J. R. Hershey, F. Weninger,
% "Sparse NMF - half-baked or well done?,"
% MERL Technical Report, TR2015-023, March 2015
% 	@TechRep{LeRoux2015mar,
% 	  author = {{Le Roux}, J. and Hershey, J. R. and Weninger, F.},
% 	  title = {Sparse {NMF} - half-baked or well done?},
% 	  institution = {Mitsubishi Electric Research Labs (MERL)},
% 	  number = {TR2015-023},
% 	  address = {Cambridge, MA, USA},
% 	  month = mar,
% 	  year = 2015
% 	}
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%   Copyright (C) 2015 Mitsubishi Electric Research Labs (Jonathan Le Roux,
%                                      Felix Weninger, John R. Hershey) 
%   Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0) 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if ~exist('params', 'var') 
    params = struct;
end

if ~isfield(params, 'max_iter') 
    params.max_iter = 100;
end
if ~isfield(params, 'random_seed') 
    params.random_seed = 1;
end

if ~isfield(params, 'conv_eps') 
    params.conv_eps = 1e-3;
end
if ~isfield(params, 'cf') 
    params.cf = 'kl';
end
if ~isfield(params, 'is_mask') 
    params.is_mask = true;
end
if params.is_mask
    if ~isfield(params, 'mask') 
        error('Mask abscent!');
    else
        mask = params.mask;
    end
end
if ~isfield(params, 'learning_rate')
    learning_rate = 0;
else
    learning_rate = params.learning_rate;
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
    lambda_social = 0;
%     social_matrix = sparse(eye(size(R,2)));
    social_matrix = 1;
end
if lambda_item > 0 && isfield(params, 'item_matrix')
	item_matrix = params.item_matrix;
else
    lambda_item = 0;
%     item_matrix = sparse(eye(size(R,1)));
    item_matrix = 1;
end


%%

col_ind = sum(abs(R),1) > 0;
row_ind = sum(abs(R),2) > 0;

if sum(col_ind) < 1 || sum(row_ind) < 1
    W_return = zeros(size(R,1),dim);
    H_return = zeros(dim, size(R,2));
    iter = 0; 
    cost = 0;
    return;
end


R_original = R;
W_return = zeros(size(R_original,1), dim);
H_return = zeros(dim, size(R_original,2));


R = R(row_ind, col_ind);
if lambda_item
    item_matrix = item_matrix(row_ind, row_ind);
end
if lambda_social
    social_matrix = social_matrix(col_ind, col_ind);
end

costparam.lambda = lambda;
costparam.lambda_social = lambda_social;
costparam.lambda_item = lambda_item;
costparam.social_matrix = social_matrix;
costparam.item_matrix = item_matrix;

mask = mask(row_ind, col_ind);
m = size(R, 1);
n = size(R, 2);


%% 
if params.random_seed > 0 
    rng('default');
    rng(params.random_seed);
end
W = rand(m, dim);
H = rand(dim, n);


FLR = 1e-9;

% Normalize the columns of W and rescale H accordingly
Wn = sqrt(sum(W.^2));
Wn(Wn == 0) = FLR;
W  = bsxfun(@rdivide,W, Wn);
H  = bsxfun(@times,  H, Wn');

if ~isfield(params, 'display') 
    params.display = 0;
end


% flr = -inf;
% last_cost = Inf;


if params.is_mask
    WH = mask_result(mask, W, H);
else
    WH = W * H;
end


% tic
options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'Display','off', 'OptimalityTolerance',1e-2);

for iter = 1:params.max_iter
    % H updates, 
    
    if learning_rate >= 0
        dph = W' * WH + lambda * H;
        dmh = W' * R;
        if lambda_social > 0
            dph = dph + lambda_social * H;
            dmh = dmh + lambda_social * (H * social_matrix);
        end

        if learning_rate ~= 0
    %             fprintf("coefficient = %f", mean(mean(dmh ./ dph)));
            H = H + params.learning_rate * (dmh - dph);
        else
            dph(dph == 0) = FLR;
    %         dmh(dmh < 0) = FLR;
            H = H .* dmh ./ dph;
        end
        if params.is_mask
            WH = mask_result(mask, W, H);
        else
            WH = W * H;
        end
    else
    
    [H_change, cost] = fminunc(@(H)CostFun(W, H, R, mask, true, costparam), H, options);
    H = H_change;
    end
    
    

    % W updates
%     dpw = WH * H' + bsxfun(@times, sum((R * H' + lambda_item * (W' * item_matrix)') .* W), W) + (lambda_item + lambda) * W;
%     dmw = R * H' + lambda_item * (W' * item_matrix)' + bsxfun(@times, sum((WH * H' + (lambda_item + lambda) * W) .* W), W);

    if learning_rate >=0
        temp1 = R * H';
        temp2 = WH * H';

        dpw = WH * H';
        dmw = R * H';

        if lambda
            dpw = dpw + lambda * W; 
    %             dmw = dmw + bsxfun(@times, sum((lambda * W) .* W), W);
            temp2 = temp2 + lambda * W;
        end

        if lambda_item
            temp1 = temp1 + lambda_item * (W' * item_matrix)';
            temp2 = temp2 + lambda_item * W;

            dpw = dpw + lambda_item * W;
            dmw = dmw + lambda_item * (W' * item_matrix)';
        end

        dpw = dpw + bsxfun(@times, sum(temp1 .* W), W);
        dmw = dmw + bsxfun(@times, sum(temp2 .* W), W);

        if learning_rate ~= 0
    %                 fprintf("\tH_coefficient = %f\n", mean(mean(dmw ./ dpw)));
            W = W + params.learning_rate * (dmw - dpw);
        else
            dpw(dpw == 0) = FLR;
    %         dmw(dmw < 0) = FLR;
            W = W .* dmw ./ dpw;
        end
    else
        [W_change, cost] = fminunc(@(W)CostFun(W, H, R, mask, false, costparam), W, options);
        W = W_change;
    end

    % Normalize the columns of W
    W = bsxfun(@rdivide,W,sqrt(sum(W.^2)));
    W(isnan(W)) = 0;
    
    
    
    
    if params.is_mask
        WH = mask_result(R, W, H);
    else
        WH = W * H;
    end
    % Compute the objective function
    div = sum(sum((R - WH) .^ 2))/2;
    cost = div + ...
        lambda * (sum(sum(W .^2 ))/2 + sum(sum(H .^2 ))/2) + ...
        lambda_item * sum(sum((W - (W' * item_matrix)').^2)) +...
        lambda_social * sum(sum((H - H * social_matrix).^2)); 
    
    if params.display ~= 0
        fprintf('iteration[%d], cost = %.3e\n', iter, full(cost));
    end
    
    % Convergence check
    if iter > 1 && params.conv_eps > 0
        e = abs(last_cost - cost) / last_cost; 
        if (e < params.conv_eps)
%             disp('Convergence reached, aborting iteration')
            break;
        end
    end
	last_cost = cost;
end

W_return(row_ind,:) = W;
H_return(:, col_ind) = H;

% toc;
end

function [dp, dm] = update(A, dp, dm)
    dp = dp + positive(A);
    dm = dm + negative(A);
end


function [X_positive] = positive(X)
    X_positive = (X + abs(X))/2;
end

function [X_negative] = negative(X)
    X_negative = (-X + abs(X))/2;
end