

% BoostCF: Explainable Boost Collaborative Filtering by Leveraging Social network
% and Feature Information
%
% Written by Chongming Gao (chongming.gao@gmail.com)
%            Dept. of Computer Science and Engineering,
%            University of Electronic Science and Technology of China
%
% Reference:
%
%  [1] Chongming Gao et al. 
%      BoostCF: Explainable Boos t Collaborative Filtering by Leveraging Social network
%      and Feature Information
% 
%  [2] Sangho Suh et al.
%      Boosted L-EnsNMF: Local Topic Discovery via Ensemble of Nonnegative Matrix Factorization.
%      IEEE International Conference on Data Mining 2016.
%
%  [3] Da Kuang Haesun Park
%      Fast Rank-2 Nonnegative Matrix Factorization for Hierarchical Document Clustering
%      International conference on Knowledge Discovery and Data mining 2013
% 
%  [4] Hyunsoo Kim and Haesun Park
%      Sparse non-negative matrix factorizations via alternating 
%      non-negativity-constrained least squares for microarray data analysis
%
% Please send bug reports, comments, or questions to Chongming Gao.
% This code comes with no guarantee or warranty of any kind.
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
% [Ws, Hs, Drs, Dcs, As] = lens_nmf(A, k, topk, iter); 


function [Ws, Hs, iter,As] = boostCF(A, params)
    
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
    
    if ~isfield(params, 'total')
        total = 1000;
    else
        total = params.total;
    end
    if ~isfield(params, 'topN')
        topN = 100;
    else
        topN = params.topN;
    end
    if ~isfield(params, 'local')
        local = 3;
    else
        local = params.local;
    end
    
    if ~isfield(params, 'dim') 
        dim = 1;
    else
        dim = params.dim;
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
    Ws = cell(total, 1); 
    Hs = cell(total, 1); 
    Rs = cell(total, 1);
    Rs{1} = A;
    
%     if exist('social_matrix', 'var') && params.hasSocial
%         has_social_info = true;
%         num_user = size(A,2);
%         for i = 1:num_user
%             social_matrix(i,i) = 0;
%         end
%         social_matrix = bsxfun(@rdivide, social_matrix, sum(social_matrix,2));
%         combine_matrix = params.alpha * social_matrix + (1 - params.alpha) * eye(num_user);
%     else
%         has_social_info = false;
%     end
    
%     figure
%     imagesc(A);
%     hold off
    Original_unexplained = sum(sum(abs(A)));
%     disp("===============BoostCF=================")
    fprintf("The initial unexplained part (sum of rating matrix) is %f\n", full(Original_unexplained));
    fprintf('learningRate=[%f], dim=[%d], lambda=[%f], lambda_social=[%f], lambda_item=[%f], sim_threshold=[%f]\n', params.learning_rate, dim, lambda, lambda_social, lambda_item, params.similarity_threshold);
    fprintf("--------------------------------------------\n");
    if isfield(params,'fid')
        fprintf(params.fid, "The initial unexplained part (sum of rating matrix) is %f\n", full(Original_unexplained));
        fprintf(params.fid, 'learningRate=[%f], dim=[%d], lambda=[%f], lambda_social=[%f], lambda_item=[%f], sim_threshold=[%f]\n', params.learning_rate, dim, lambda, lambda_social, lambda_item, params.similarity_threshold);
        fprintf(params.fid, "-------------------------------------------\n");
    end
    unexplained_last = Original_unexplained;
    
%%
    
    for iter=1:(total) % loop for given number of iterations
        rng(iter);
        As = Rs{iter};
        
        if isWithSample && iter >= local
            try
                row_idx = datasample(1:size(As,1), 1, 'Replace', false, 'Weights', full(sum(abs(As),2)));
                col_idx = datasample(1:size(As,2), 1, 'Replace', false, 'Weights', full(sum(abs(As),1)));
            catch
                fprintf( "Sample fail, the weight has some NaN, break!\n");
                fprintf( "-------------------------------------------\n");
                if isfield(params,'fid')
                    fprintf(params.fid, "Sample fail, the weight has some NaN, break!\n");
                    fprintf(params.fid, "-------------------------------------------\n");
                end
                break;
            end
            [A_cossim_row, A_cossim_col] = get_cosine_similarity(As, row_idx, col_idx);
            
            
            [As, subsize_row, subsize_col, row_ind, col_ind, item_ind, social_ind] = sample_RowandCol(As, A_cossim_row, A_cossim_col, params.similarity_threshold, sampleThreshold, row_idx, col_idx, has_social_network, has_item_network, social_matrix, item_matrix,topN);
        else
            subsize_row = size(A,1);
            subsize_col = size(A,2);
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

        if iter >= local
            param_MF = params;
        else
            param_MF = params;
            param_MF.learning_rate = 0;
        end
        
        [W, H, iteration] = MF(As, dim, param_MF);
        
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
                Rs{iter+1} = update_res_matrix(Rs{iter}, Ws{iter},Hs{iter}, mask); 
            else
                Rs{iter+1} = update_res_matrix(Rs{iter}, Ws{iter},Hs{iter});
            end
%             figure
%             imagesc(Rs{iter+1});
%             hold off
            unexplained = sum(sum(abs(Rs{iter + 1})));
            percentage = unexplained/Original_unexplained;
            
            fprintf("Round[%d]: Size[%d×%d], Iteration[%d], Unexplained part[%f], Percentage[%f%%], delta%%=[%f%%]\n", ...
                iter, full(subsize_row), full(subsize_col), iteration, full(unexplained), full(percentage) * 100, full((unexplained_last - unexplained)/Original_unexplained * 100));
            if isfield(params,'fid')
                fprintf(params.fid, "Round[%d]: Size[%d×%d], Iteration[%d], Unexplained part[%f], Percentage[%f%%], delta%%=[%f%%]\n", ...
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
    if isfield(params,'fid')
        fprintf(params.fid, '\n\n');
%         fprintf(params.fid, 'dim=[%d], lambda=[%f], lambda_social=[%f], lambda_item=[%f]\n', dim, lambda, lambda_social, lambda_item);
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

function [newA, subsize_row, subsize_col, row_indicate, col_indicate, item_indicate, social_indicate] = sample_RowandCol(A, A_cossim_row, A_cossim_col, threshold, sampleThreshold, row_idx, col_idx, has_social_network, has_item_network, social_matrix, item_matrix,topN)
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
    
    if subsize_row <= sampleThreshold || subsize_row > topN
%         row_indicate = (1:size(A,1))';
%         subsize_row = size(A,1);
        [~, row_indicate] = maxk(A_cossim_row,topN);
        subsize_row = topN;
    end

    if subsize_col <= sampleThreshold || subsize_col > topN
%         col_indicate = (1:size(A,2))';
%         subsize_col = size(A,2);
        [~, col_indicate] = maxk(A_cossim_col,topN);
        subsize_col = topN;
    end
    
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

