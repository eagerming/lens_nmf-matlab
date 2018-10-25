function [w, h] = boostMF(R, params)
    %% Initialize all the parameters
    
    col_ind = sum(R,1) > 0;
    row_ind = sum(R,2) > 0;
    R_original = R;
    R = R(row_ind, col_ind);
    
    num_row = size(R, 1);
    num_col = size(R, 2);
    
    if ~exist('params', 'var') 
        params = struct;
    end

    if ~isfield(params, 'max_iter') 
        params.max_iter = 1000;
    end
    if ~isfield(params, 'random_seed')
        params.random_seed = 1;
    end
%     if ~isfield(params, 'sparsity')
%         params.sparsity = 0;
%     end
    if ~isfield(params, 'conv_eps') 
        params.conv_eps = 1e-4;
    end

    if ~isfield(params, 'is_zero_mask_of_missing')
        params.is_zero_mask_of_missing = true;
    end
    if ~isfield(params, 'display') 
        params.display = 0;
    end
    
    
    if params.random_seed > 0 
        rng(params.random_seed);
    end
    
    %% Initialize W and H 
    if ~isfield(params, 'init_w') 
        if ~isfield(params, 'dim')
            error('Number of components or initialization must be given') 
        end
        dim = params.dim;
        w = rand(num_row, dim); 
    else
        ri = size(params.init_w, 2);
        w(:, 1:ri) = params.init_w;
        if isfield(params, 'dim') && ri < params.dim
            w(:, (ri + 1) : params.dim) = rand(num_row, params.dim - ri);
            dim = params.dim;
        else
            dim = ri;
        end
    end
    
    if ~isfield(params, 'init_h') 
        if learning_rate == 0
            h = rand(dim, num_col);
        else
            h = zeros(dim, num_col);
        end
    elseif ischar(params.init_h) && strcmp(params.init_h, 'ones') 
        fprintf('sup_nmf: Initalizing H with ones.\n');
        h = ones(dim, num_col);
    else
        h = params.init_h;
    end

    % Normalize the columns of W and rescale H accordingly
    wn = sqrt(sum(w.^2));
    w  = bsxfun(@rdivide, w, wn);
    h  = bsxfun(@times, h, wn');
    
    
    %% 
    
    if params.is_zero_mask_of_missing
        lambda = mask_result(R, w, h);
    else
        lambda = w * h;
    end
    
    flr = 1e-4;
    
    for it = 1:params.max_iter
    
        % H updates
        dph = w' * lambda;
        dmh = w' * R;
        if params.learning_rate ~= 0
%             fprintf("coefficient = %f", mean(mean(dmh ./ dph)));
            h = h + params.learning_rate * (dmh - dph);
        else
            dph(dph == 0) = flr;
            h = h .* dmh ./ dph;
        end
                
        if params.is_zero_mask_of_missing
            lambda = mask_result(R, w, h);
        else
            lambda = w * h;
        end


        %% W updates
        dpw = lambda * h';
        dmw = R * h';
        if learning_rate ~= 0
%             fprintf("\tH_coefficient = %f\n", mean(mean(dmw ./ dpw)));
            w = w + params.learning_rate * (dmw - dpw);
        else
            dpw(dpw == 0) = flr;
            w = w .* dmw ./ dpw;
        end

        
        % Normalize the columns of W and rescale H accordingly
        wn = sqrt(sum(w.^2));
        w  = bsxfun(@rdivide, w, wn);
        h  = bsxfun(@times, h, wn');
        
        
        if params.is_zero_mask_of_missing
            lambda = mask_result(v, w, h);
        else
            lambda =w * h;
        end
    end
        


        %% Compute the objective function
        R_approximate = mask_result(R, w, h);

        cost = div + sum(sum(params.sparsity .* h));

        objective.div(it)  = div;
        objective.cost(it) = cost;

        if params.display ~= 0
            fprintf('iteration %d div = %.3e cost = %.3e\n', it, full(div), full(cost));
        end

        % Convergence check
        if it > 1 && params.conv_eps > 0
            e = abs(cost - last_cost) / last_cost; 
            if (e < params.conv_eps)
    %             disp('Convergence reached, aborting iteration') 
                objective.div = objective.div(1:it); 
                objective.cost = objective.cost(1:it);
                break
            end
        end
        last_cost = cost;
    end
    

end




