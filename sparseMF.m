function [W, H] = sparseMF(R, params)
    %% Initialize all the parameters
    num_row = size(R, 1);
    num_col = size(R, 2);
    
    if ~exist('params', 'var') 
        params = struct;
    end

    if ~isfield(params, 'max_iter') 
        params.max_iter = 100;
    end
    if ~isfield(params, 'random_seed') 
        params.random_seed = 1;
    end
    if ~isfield(params, 'sparsity') 
        params.sparsity = 0;
    end
    if ~isfield(params, 'conv_eps') 
        params.conv_eps = 1e-5;
    end

    if ~isfield(params, 'is_mask')
        params.is_mask = true;
    end
    if ~isfield(params, 'display') 
        params.display = 0;
    end
    if ~isfield(params, 'method') 
        params.method = 'DALM';
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
    methodName = ['Solve' params.method] ;
    methodHandle = str2func(methodName) ; 
    
    for it = 1:params.max_iter
    
        % H updates
        for i = 1:num_col
            b = R(:,i);
            if params.is_mask
                mask = b>0;
                b = b(mask);
                A = w(mask,:);
            else
                A = w;
            end
            
              
            [x, nIter, curTimeEst] = methodHandle(A, b);
        
            h(:, i) = x;
        end
        
        %% W updates
        for j = 1:num_row
            b = R(j, :)';
            if params.is_mask
                mask = b > 0;
                b = b(mask);
                A = h(:,mask)';
            else
                A = h';
            end
        
            [x, nIter, curTimeEst] = methodHandle(A, b) ;
            w(j, :) = x';
        end
        
        % Normalize the columns of W and rescale H accordingly
        wn = sqrt(sum(w.^2));
        w  = bsxfun(@rdivide, w, wn);
        h  = bsxfun(@times, h, wn');
        


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




