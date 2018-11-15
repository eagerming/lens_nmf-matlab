

% Collaborative Filtering with Social Local Models
% Paper author: Huan Zhao et al.
% Code Written by Chongming Gao (chongming.gao@gmail.com)
%            Dept. of Computer Science and Engineering,
%            University of Electronic Science and Technology of China
%
% Last modified 10/26/2018
%
% <Inputs>
%
%        A : Input matrix 
%        k : Number of dimensions per iteration
%        total : Maximal dimension dimension
%        iter : Number of iterations
%        beta : coefficient for sparsity control. 
%               Larger beta generates higher sparseness on H.
%               Too large beta is not recommended.
%        isWithSample : If true, resample and reconstruct the residue matrix
%
% <Outputs>
%
%        Ws, Hs: Results of rank-2 NMF
%        Drs : Set of stage-wise rows with cosine similarity values 
%        Dcs : Set of stage-wise columns with cosine similarity values 
%        As : Updated input matrix
% 
% <Usage Example> 
%


function [Ws, Hs, A_sloma, iter] = SLOMA(A, params)
    
    if ~isfield(params, 'is_mask') 
        if isfield(params, 'mask') 
            params.is_mask = true;
        else
            params.is_mask = false;
        end
    end
    if params.is_mask
        mask = params.mask;
    end
    
    if ~isfield(params, 'similarity_threshold') 
        params.similarity_threshold = 0;
    end
    if ~isfield(params, 'isWithSample')
        isWithSample = true;
    else
        isWithSample = params.isWithSample;
    end
    
    numOfBlock = params.numOfBlock;
    
    if ~isfield(params, 'dim_sloma') 
        dim_sloma = 1;
    else
        dim_sloma = params.dim_sloma;
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
    
    has_social_network = isfield(params, 'social_matrix');
    has_item_network = isfield(params, 'item_matrix');
    if has_social_network
        social_matrix = params.social_matrix;
    else
        lambda_social = 0;
        social_matrix = [];
    end
    if has_item_network 
        item_matrix = params.item_matrix;
    else
        lambda_item = 0;
        item_matrix = [];
    end
    
    if isfield(params, 'sampleThreshold')
        sampleThreshold = params.sampleThreshold;
    else
        sampleThreshold = 0;
    end
    if isfield(params, 'isVisual') && params.isVisual && size(A,1) == 2
        isVisual = true;
        last_endPoint = zeros(size(A));
    else
        isVisual = false;
    end
    
    

    
%%    
    % Initialization.
    
    Original_unexplained = sum(sum(abs(A)));
%     disp("===============BoostCF=================")
    fprintf("[SLOMA !!!!!]The initial unexplained part (sum of rating matrix) is %f\n", full(Original_unexplained));
    fprintf('dim=[%d], lambda=[%.2f], lambda_social=[%.2f], lambda_item=[%.2f], sim_threshold=[%f]\n', dim_sloma, lambda, lambda_social, lambda_item, params.similarity_threshold);
    fprintf("--------------------[SLOMA !!!!!]------------------------\n");
    if isfield(params,'fid')
        fprintf(params.fid, "The initial unexplained part (sum of rating matrix) is %f\n", full(Original_unexplained));
        fprintf(params.fid, 'dim=[%d], lambda=[%.2f], lambda_social=[%.2f], lambda_item=[%.2f], sim_threshold=[%f]\n', dim_sloma, lambda, lambda_social, lambda_item, params.similarity_threshold);
        fprintf(params.fid, "--------------------[SLOMA !!!!!]-----------------------\n");
    end
    unexplained_last = Original_unexplained;
    
%%
    Ws = cell(numOfBlock, 1); 
    Hs = cell(numOfBlock, 1); 
    Rs = cell(numOfBlock + 1, 1);
    Rs{1} = A;

    weight_row_init = full(sum(abs(A),2));
    weight_row = weight_row_init;
    weight_col_init = full(sum(abs(A),1));
    weight_col = weight_col_init;
    
    A_block = cell(numOfBlock,1);
    rowSeed_list = zeros(numOfBlock,1);
    colSeed_list = zeros(numOfBlock,1);
    row_ind_list = zeros(size(A,1),numOfBlock);
    col_ind_list = zeros(size(A,2),numOfBlock);
    index_vec_list = zeros(numel(A),numOfBlock);
    
    A_sloma = zeros(size(A));
    
    for iter=1:(numOfBlock) % loop for given number of iterations
        rng(iter);
        try
            row_idx = datasample(1:size(A,1), 1, 'Replace', false, 'Weights', weight_row);
            col_idx = datasample(1:size(A,2), 1, 'Replace', false, 'Weights', weight_col);
        catch
            fprintf( "Sample fail, the weight has some NaN, break!\n");
            fprintf( "---------------------[SLOMA !!!!!]----------------------\n");
            if isfield(params,'fid')
                fprintf(params.fid, "Sample fail, the weight has some NaN, break!\n");
                fprintf(params.fid, "------------------[SLOMA !!!!!]-------------------------\n");
            end
            break;
        end
        [A_cossim_row, A_cossim_col] = get_cosine_similarity(A, row_idx, col_idx);
        [A_i, subsize_row, subsize_col, row_ind, col_ind, item_ind, social_ind] = sample_RowandCol(A, A_cossim_row, A_cossim_col, params.similarity_threshold, sampleThreshold, row_idx, col_idx, has_social_network, has_item_network, social_matrix, item_matrix);
        A_block{iter} = A_i;
        rowSeed_list(iter) = row_idx;
        colSeed_list(iter) = col_idx;
        row_ind_list(:,iter) = row_ind;
        col_ind_list(:,iter) = col_ind;
        
        index = zeros(size(A));
        index(row_ind, col_ind) = 1;
        index_vec_list(:,iter) = index(:);
        
        weight_row(row_ind) = 0;
        weight_col(col_ind) = 0;
        
        if sum(weight_row) <= eps
            weight_row = weight_row_init;
        end
        if sum(weight_col) <= eps
            weight_col = weight_col_init;
        end
        
        %% Try to use SparseNM
        % profile on  
        
%         if dim > 3
        
%         dim_now = dim;
%         while true
%             [W, H, isSuccess, iteration] = boostNMF(As, dim_now, params);
%             if isSuccess
%                 break
%             end
%             dim_now = dim_now - 1;
%         end

%         params.r = dim;
%         params.cf = 'ed';
%         [W, H] = sparse_nmf(As, params);

        [W, H, iteration] = MF(A_i, dim_sloma, params);
        A_sloma = A_sloma + W * H;
        
        RR = A_sloma(:);
        times = max(sum(index_vec_list,2), 1e-6);
        RR = RR ./ times;
        RR = reshape(RR, size(A,1), size(A,2));
        
        % profile viewer/
        % p = profile('info')
        % s = profile('status')
        
        Ws{iter} = W;
        Hs{iter} = H;
        if isVisual
            last_endPoint = visualBasicVector(W,H,last_endPoint);
        end
        
        %% 
%         if has_social_info
%             Hs{iter} = (combine_matrix * Hs{iter}')';
%         end

%         if iter <= total
         
            % fix W and use unweighted version of A to get H            
            
            %[Hs{iter},temp,suc_H,numChol_H,numEq_H] = nnlsm_activeset(Ws{iter}'*Ws{iter},Ws{iter}'*A,0,1,bsxfun(@times,Hs{iter}',1./Dcs{iter})');
            % Sangho Suh writed as above, revised by Chongming Gao. The
            % following code can also be truncated.
%             [Hs{iter},temp,suc_H,numChol_H,numEq_H] = nnlsm_activeset(Ws{iter}'*Ws{iter},Ws{iter}'*Rs{iter},0,1,bsxfun(@times,Hs{iter}',1./Dcs{iter})');

            % update residual matrix
            
            
            if params.is_mask
                Rs{iter+1} = (A - RR) .* mask; 
            else
                Rs{iter+1} = (A - RR);
            end
%             figure
%             imagesc(Rs{iter+1});
%             hold off
            unexplained = sum(sum(abs(Rs{iter + 1})));
            percentage = unexplained/Original_unexplained;
            
            fprintf("[SLOMA !!!!!] Round[%d]: Size[%d×%d], Iteration[%d], Unexplained part[%f], Percentage[%f%%], delta%%=[%f%%]\n", ...
                iter, full(subsize_row), full(subsize_col), iteration, full(unexplained), full(percentage) * 100, full((unexplained_last - unexplained)/Original_unexplained * 100));
            if isfield(params,'fid')
                fprintf(params.fid, "[SLOMA !!!!!] Round[%d]: Size[%d×%d], Iteration[%d], Unexplained part[%f], Percentage[%f%%], delta%%=[%f%%]\n", ...
                iter, full(subsize_row), full(subsize_col), iteration, full(unexplained), full(percentage) * 100, full((unexplained_last - unexplained)/Original_unexplained * 100));
            end
%             if isfield(params,'exitAtDeltaPercentage')
%                 if abs(unexplained_last - unexplained)/Original_unexplained < params.exitAtDeltaPercentage ||...
%                           unexplained/Original_unexplained < params.exitAtDeltaPercentage
%                     break;
%                 end
%             end

            unexplained_last = unexplained;
%         end
            
        
    end
    A_sloma = A_sloma(:);
    times = max(sum(index_vec_list,2), 1e-6);
    A_sloma = A_sloma ./ times;
    A_sloma = reshape(A_sloma, size(A,1), size(A,2));
    
    if isfield(params,'fid')
        fprintf(params.fid, '\n\n');
%         fprintf(params.fid, 'dim=[%d], lambda=[%.2f], lambda_social=[%.2f], lambda_item=[%.2f]\n', dim, lambda, lambda_social, lambda_item);
%         fprintf(params.fid, "Terminate at Round [%d]: Unexplained part: %f, Percentage %f%%, delta%%=%f\n", ...
%                 iter, full(unexplained), full(percentage) * 100, full((unexplained_last - unexplained)/Original_unexplained));
    end
    
end

%%
% function [Dr_new, Dc_new] = getWeight(A,A_cossim_row,A_cossim_col,trm_idx,doc_idx)
% 
%     row_smooth = .01;
%     col_smooth = .01;
% 
%     Dr_new = A_cossim_row(:,trm_idx)*(1-row_smooth)+row_smooth;
%     Dc_new = A_cossim_col(:,doc_idx)*(1-col_smooth)+col_smooth;
%     
% end

%%

function [newA, subsize_row, subsize_col, row_indicate, col_indicate, item_indicate, social_indicate] = sample_RowandCol(A, A_cossim_row, A_cossim_col, threshold, sampleThreshold, row_idx, col_idx, has_social_network, has_item_network, social_matrix, item_matrix)
    row_indicate = A_cossim_row >= threshold;
    col_indicate = A_cossim_col >= threshold;
    item_indicate = [];
    social_indicate = [];
    if has_social_network
        social_indicate = social_matrix(:,col_idx) > 0;
        col_indicate(social_indicate) = true;
    end
    if has_item_network
        item_indicate = item_matrix(:,row_idx) > 0;
        row_indicate(item_indicate) = true;
    end
    
    
    subsize_row = sum(row_indicate);
    subsize_col = sum(col_indicate);
    
%     if subsize_row <= sampleThreshold
    if true
        row_indicate = (1:size(A,1))';
        subsize_row = size(A,1);
    end
    
%     if subsize_col <= sampleThreshold
%         col_indicate = (1:size(A,2))';
%         subsize_col = size(A,2);
%     end
    
    newA = sparse(size(A,1),size(A,2));
    newA(row_indicate,col_indicate) = A(row_indicate,col_indicate);
end


function [A_cossim_row, A_cossim_col] = get_cosine_similarity(A, row_idx, col_idx)
    
    flr = 1e-6;
    A_l2norm_row = bsxfun(@rdivide,A',max(flr, sqrt(sum((A').^2))))';
%     A_l2norm_row = A;
    A_cossim_row = A_l2norm_row * A_l2norm_row(row_idx,:)'; 

    A_l2norm_col = bsxfun(@rdivide,A, max(flr, sqrt(sum(A.^2))));
%     A_l2norm_col = A;
    A_cossim_col = A_l2norm_col' * A_l2norm_col(:,col_idx);
    
end

% 
% function [newA] = update(A, Dr, Dc)
% 
%     newA = bsxfun(@times,A, Dr);  % multiply each row of A with Dr row
%     newA = bsxfun(@times,newA',Dc)';  % multiply each column of A with Dc column  
%     
% end

%%
function [newA] = update_res_matrix(A, W, H, mask)    
    num_row = size(A,1);
    num_col = size(A,2);
    
    if exist('mask','var')  
        WH = sparse(num_row, num_col);
        [ii, jj] = find(mask);
        for k = 1:length(ii)
            WH(ii(k),jj(k)) = W(ii(k), :) * H(:, jj(k));
        end
    else
        WH = W * H;
    end
    
    newA = A - WH;
    
end
