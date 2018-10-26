

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


function [Ws, Hs, iter,As] = boostCF(A, params, social_matrix)
    
    if ~isfield(params, 'is_zero_mask_of_missing') 
        params.is_zero_mask_of_missing = true;
    end
    if params.is_zero_mask_of_missing
        mask = A~=0;
        params.mask = mask;
    end
    
    if ~isfield(params, 'similarity_threshold') 
        params.similarity_threshold = 0.5;
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
    if isfield(params, 'sampleThreshold')
        sampleThreshold = params.sampleThreshold;
    else
        sampleThreshold = 1;
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
    Original_unexplained = sum(sum(A));
    disp("===============BoostCF=================")
    fprintf("The initial unexplained part (sum of rating matrix) is %f\n", full(Original_unexplained));
    fprintf('dim=[%d], lambda=[%.2f], lambda_social=[%.2f], lambda_item=[%.2f]\n', dim, lambda, lambda_social, lambda_item);
    unexplained_last = Original_unexplained;
    
%%
    for iter=1:(total) % loop for given number of iterations

        As = Rs{iter};
        
        if isWithSample
            
            row_idx = datasample(1:size(As,1), 1, 'Replace', false, 'Weights', full(sum(abs(As),2)));
            col_idx = datasample(1:size(As,2), 1, 'Replace', false, 'Weights', full(sum(abs(As))));
            [A_cossim_row, A_cossim_col] = get_cosine_similarity(As, row_idx, col_idx);
            
            
            [As, subsize_row, subsize_col] = sample_RowandCol(As, A_cossim_row, A_cossim_col, params.similarity_threshold, sampleThreshold);
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

        [W, H, iteration] = MF(As, dim, params);
        
        % profile viewer/
        % p = profile('info')
        % s = profile('status')
        
        Ws{iter} = W;
        Hs{iter} = H;
        
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
            if params.is_zero_mask_of_missing
                Rs{iter+1} = update_res_matrix(Rs{iter}, Ws{iter},Hs{iter}, mask); 
            else
                Rs{iter+1} = update_res_matrix(Rs{iter}, Ws{iter},Hs{iter});
            end
%             figure
%             imagesc(Rs{iter+1});
%             hold off
            unexplained = sum(sum(abs(Rs{iter + 1})));
            percentage = unexplained/Original_unexplained;
            
            fprintf("Round[%d]: Size[%d×%d], Iteration[%d], Unexplained part[%f], Percentage[%f]%%\n", ...
                iter, full(subsize_row), full(subsize_col), iteration, full(unexplained), full(percentage) * 100);
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
        fprintf(params.fid, 'dim=[%d], lambda=[%.2f], lambda_social=[%.2f], lambda_item=[%.2f]\n', dim, lambda, lambda_social, lambda_item);
        fprintf(params.fid, "Terminate at Round [%d]: Unexplained part: %f, Percentage %f%%, delta%%=%f\n", ...
                iter, full(unexplained), full(percentage) * 100, full((unexplained_last - unexplained)/Original_unexplained));
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

function [newA, subsize_row, subsize_col] = sample_RowandCol(A, A_cossim_row, A_cossim_col, threshold, sampleThreshold)
    row = A_cossim_row >= threshold;
    col = A_cossim_col >= threshold;
    
    subsize_row = sum(row);
    subsize_col = sum(col);
    
    if subsize_row <= sampleThreshold
        row = 1:size(A,1);
        subsize_row = size(A,1);
    end
    if subsize_col <= sampleThreshold
        col = 1:size(A,2);
        subsize_col = size(A,2);
    end
    
    newA = sparse(size(A,1),size(A,2));
    newA(row,col) = A(row,col);
end


function [A_cossim_row, A_cossim_col] = get_cosine_similarity(A, row_idx, col_idx)
    
    flr = 1e-6;
    A_l2norm_row = bsxfun(@rdivide,A',max(flr, sqrt(sum((A').^2))))';
    A_cossim_row = A_l2norm_row * A_l2norm_row(row_idx,:)'; 

    A_l2norm_col = bsxfun(@rdivide,A, max(flr, sqrt(sum(A.^2))));
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

