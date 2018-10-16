

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


function [Ws, Hs, iter,As] = boostCF(A, param, social_matrix)

    isWithSample = param.isWithSample;
    total = param.total;
    dim = param.dim;
    
    if isWithSample
        % apply l2-normalization and get row-wise and column-wise cosine similarity values 
        
        A_l2norm_row = bsxfun(@rdivide,A',sqrt(sum((A').^2)))';
        A_cossim_row = A_l2norm_row*A_l2norm_row'; 

        A_l2norm_col = bsxfun(@rdivide,A,sqrt(sum(A.^2)));
        A_cossim_col = A_l2norm_col'*A_l2norm_col;
        
        Drs = []; Dcs = [];
    end
    
    
%%    
    % Initialization.
    Ws = cell(total, 1); 
    Hs = cell(total, 1); 
    Rs = cell(total, 1);
    As = A; 
    Rs{1} = A;
    
    % Parameters Setting.
    
    param.vec_norm = 2.0;
    param.normW = true;
    param.tol = 1e-4;

    if exist('social_matrix', 'var') && param.hasSocial
        has_social_info = true;
        num_user = size(A,2);
        for i = 1:num_user
            social_matrix(i,i) = 0;
        end
        social_matrix = bsxfun(@rdivide, social_matrix, sum(social_matrix,2));
        combine_matrix = param.alpha * social_matrix + (1 - param.alpha) * eye(num_user);
    else
        has_social_info = false;
    end
    
%     figure
%     imagesc(A);
%     hold off
    Original_unexplained = sum(sum(A));
    disp("===============BoostCF=================")
    fprintf("The initial unexplained part (sum of rating matrix) is %f\n", full(Original_unexplained));
    unexplained_last = Original_unexplained;
    
%%
    for iter=1:(total) % loop for given number of iterations

        if isWithSample
            if iter == 1   
                row_idx = datasample(1:size(A,1),1,'Replace',false);
            else
                % sample with weight to get one row index
                row_idx = datasample(1:size(A,1),1,'Replace',false,'Weights', sum(cell2mat(Ws'),2));
            end            
            % get one column index
            col_idx = datasample(1:size(A,2),1,'Replace',false,'Weights', full(sum(abs(Rs{iter}))));
            
            % update Drs, Dcs with cosine similarity
            [Drs{iter}, Dcs{iter}] = getWeight(Rs{iter},A_cossim_row,A_cossim_col,row_idx,col_idx);
            % update A matrix using Drs,Dcs
            As = update(Rs{iter}, Drs{iter}, Dcs{iter}); 
        else
            As = Rs{iter};
        end

%         [Ws{iter}, Hs{iter}] = boostNMF(As, dim, param);
        %% Try to use SparseNM
        % profile on  
        param.r = dim;
        [W, H, objective] = sparse_nmf(As, param);
        % profile viewer/
        % p = profile('info')
        % s = profile('status')
        
        Ws{iter} = W;
        Hs{iter} = H;
        
        %% 
        if has_social_info
            Hs{iter} = (combine_matrix * Hs{iter}')';
        end

%         if iter <= total
         
            % fix W and use unweighted version of A to get H            
            
            %[Hs{iter},temp,suc_H,numChol_H,numEq_H] = nnlsm_activeset(Ws{iter}'*Ws{iter},Ws{iter}'*A,0,1,bsxfun(@times,Hs{iter}',1./Dcs{iter})');
            % Sangho Suh writed as above, revised by Chongming Gao. The
            % following code can also be truncated.
%             [Hs{iter},temp,suc_H,numChol_H,numEq_H] = nnlsm_activeset(Ws{iter}'*Ws{iter},Ws{iter}'*Rs{iter},0,1,bsxfun(@times,Hs{iter}',1./Dcs{iter})');

            % update residual matrix 
            Rs{iter+1} = update_res_matrix(Rs{iter}, Ws{iter},Hs{iter}); 
%             figure
%             imagesc(Rs{iter+1});
%             hold off
            unexplained = sum(sum(Rs{iter + 1}));
            percentage = unexplained/Original_unexplained;
            
            fprintf("Round [%d]: Unexplained part: %f, Percentage %f%%\n", ...
                iter, full(unexplained), percentage * 100);
            if isfield(param,'exitAtDeltaPercentage')
                if (unexplained_last - unexplained)/Original_unexplained < param.exitAtDeltaPercentage
                    break;
                end
            end
            
            unexplained_last = unexplained;
            
%         end
            
    end
    if isfield(param,'fid')
        fprintf(param.fid, "Terminate at Round [%d]: Unexplained part: %f, Percentage %f%%, delta%%=%f\n", ...
                iter, full(unexplained), percentage * 100, (unexplained_last - unexplained)/Original_unexplained);
    end
    
end

%%
function [Dr_new, Dc_new] = getWeight(A,A_cossim_row,A_cossim_col,trm_idx,doc_idx)

    row_smooth = .01;
    col_smooth = .01;

    Dr_new = A_cossim_row(:,trm_idx)*(1-row_smooth)+row_smooth;
    Dc_new = A_cossim_col(:,doc_idx)*(1-col_smooth)+col_smooth;
    
end

%%
function [newA] = update(A, Dr, Dc)

    newA = bsxfun(@times,A,Dr);  % multiply each row of A with Dr row
    newA = bsxfun(@times,newA',Dc)';  % multiply each column of A with Dc column  
    
end

%%
function [newA] = update_res_matrix(A, W, H)

    newA = A - W*H;         % get residual matrix   
    newA (newA<0) = 0;      % set any negative element to zero

end

