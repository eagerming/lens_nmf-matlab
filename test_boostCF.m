% Topic Modeling Experiment 
% 
% Methods: StandardNMF, Sparse NMF, Orthogonal NMF, LDA, L-EnsNMF 
% 
% Written by Sangho Suh (sh31659@gmail.com)
%            Dept. of Computer Science and Engineering,
%            Korea University
% 
% Reference: 
% 
% [1] Sangho Suh, Jaegul Choo, Joonseok Lee and Chandan K. Reddy. 
%     L-EnsNMF: Boosted Local Topic Discovery via Ensemble of Nonnegative Matrix Factorization.
%
% [2] H. Kim and H. Park. Sparse non-negative matrix factorizations via
%     alternating non-negativity-constrained least squares for microarray data
%     analysis.
%
% [3] H. Kim and H. Park. Nonnegative matrix factorization based on
%     alternating nonnegativity constrained least squares and active set method.
%     SIAM Journal on Matrix Analysis and Applications, 30(2):713?730,
%     2008.
% 
% [4] D. Kuang and H. Park. Fast rank-2 nonnegative matrix factorization
%     for hierarchical document clustering. In Proc. the ACM SIGKDD
%     International Conference on Knowledge Discovery and Data Mining
%     (KDD), pages 739?747, 2013.
%
% [5] D. M. Blei, A. Y. Ng, and M. I. Jordan. Latent dirichlet allocation.
%     Journal of Machine Learning Research (JMLR), 3:993?1022, 2003.
%
% [6] https://github.com/kimjingu/nonnegfac-matlab
% [7] http://www.cc.gatech.edu/hpark/software/nmf bpas.zip
% [8] http://davian.korea.ac.kr/myfiles/list/Codes/orthonmf.zip
% [9] http://psiexp.ss.uci.edu/research/programs data/toolbox.html
% 
% Please send bug reports, comments, or questions to Sangho Suh.
% This comes with no guarantee or warranty of any kind.
%
% Last modified 11/04/2016
%
% 

% clear work space
clear;
close all;

% add path to library folder
addpath('./library');
addpath('./library/nmf');
addpath('./library/lens_nmf');
addpath('./library/ramkis');
addpath('./library/topictoolbox');

% add path to dataset folder
addpath('./data');

% add path to evaluation folder
addpath('./evaluation/topic_coherence')
addpath('./evaluation/total_document_coverage')




%%
% Specify number of experiment(s) to perform
loop = 1;

tic

for numOfLoop=1:loop
    
    loop = loop + 1; % increase count
    
	for choice=1:1  % value of choice belongs to [1,5] where the value indicates dataset
        
        close all;
        clearvars -except loop choice;
        dataname = {};

        
        %% Decide dataset
        
        if(choice==1)                                                               
            dataname = 'filmtrust_BoostMF';
            try
                load filmtrust;
            catch
                rating_filePath = 'data/filmtrust/rating/ratings.txt';
                trust_filePath = 'data/filmtrust/trust/trust.txt';
                [R,S,map] = loadData(rating_filePath, trust_filePath, dataname);
                clear rating_filePath trust_filePath;
            end
            has_trust = true;
            has_itemfeature = false;
        elseif(choice==2)
            load enron_tdm_n2000;
            has_dict = 1;
        end
        
        log_name = dataname;
        [fid, fidpath] = create_new_log(log_name);
            
      %% initialize

        
        
        dim = 2;   % number of topics per stage in L-EnsNMF
        stage = 8; % number of stages in L-EnsNMF
        total_topic = dim * stage; % number of total topics
        beta = 0.4;

        topk = 10; % number of top keywords to be listed in order within a topic (denoted as c1 in experiment section)

        markers = '.ox+*sdv^<>ph'; % markers for graphs
        mcnt = 0;                  % method count
        mname = {};                % method name
        speed = {};                % computing time

%         R = A;
        R_original = R;     
        
        

      %% Normalization

        % l2 normalization
        R_l2norm = bsxfun(@rdivide,R,sqrt(sum(R.^2)));
        % l1 normalization
        R_l1norm = bsxfun(@rdivide,R,sum(R));
        % tf-idf
        R_idf = tfidf2(R);
        % tf-idf & l2 normalization
        R_l2norm_idf = bsxfun(@rdivide,R,sqrt(sum(R_idf.^2)));
        % 0-1 normalization
        R_01 = (R - min(R(:))) / (max(R(:)) - min(R(:)));

        

%         R = R_l1norm;
%         R = R_idf;
%         R = R_l2norm_idf;
%         R = R_01;

        %% Splitting
        
        ratio = 0.8;
        israndom = true;
        [train_matrix, test_matrix] = rating_splitter(R, ratio, israndom);
        
        R = train_matrix;
        target_R = R;
        
        
        %% Set the dictionary
        
        if ~exist('dict_new','var') % if there is no variable called 'dict_new', then execute the following
            if exist('dictionary','var')
                dict_new = strtrim(mat2cell (dictionary, ones (size(dictionary,1),1) , size(dictionary,2) ) );
            else
                for i = 1 : size(R,1)
                    dict_new{i,1} = num2str(i);
                end
            end
        end

        if exist('titles','var') % if variable called 'titles' exist then execute the following
            % create empty cell
            title_new = cell(length(titles),1);
            % remove spaces within string
            for i=1:length(titles)
                title_new{i} = strtrim(titles{i}(2,:));
            end
        end

        
        %% standard NMF (1st method)
        mcnt = mcnt + 1; mname{mcnt} = 'StandardNMF'
        tic
        [V{mcnt},U{mcnt}] = nmf(target_R, total_topic);
        speed{mcnt} = toc;


      %% NMF + sparse (2nd method)

%         mcnt = mcnt + 1; mname{mcnt} = 'NMF_Sparse'
%         param_NMFS = [-1 beta];
% 
%         tic
%         [V{mcnt},U{mcnt}] = nmfsh_comb(target_R, total_topic, param_NMFS); 
%         speed{mcnt} = toc;
% 
%         U{mcnt} = U{mcnt}; % store transposed version of H
%         V{mcnt} = V{mcnt}; % store transposed version of W

        
        
        
      %% Sparse NMF 
      
        mcnt = mcnt + 1; mname{mcnt} = 'SparseNMF'
        param_snmf.r = total_topic;
        param_snmf.cf = 'ed';
        param_snmf.conv_eps = 1e-4;
        param_snmf.sparsity = beta;
        param_snmf.max_iter = 100;
        % param.display = 1;
        [V{mcnt},U{mcnt}] = sparse_nmf(target_R, param_snmf);
        

      %% weak ortho NMF (3rd method)
% 
%         mcnt = mcnt + 1; mname{mcnt} = 'OrthoNMF' 
% 
%         Winit=rand(size(target_R,1),total_topic); % create a matrix the size of 'size(A,1) x k_std' as W
%         Winit=Winit./repmat(sqrt(sum(Winit.^2,1)),size(target_R,1),1);  % normalize
%         Hinit=rand(total_topic,size(target_R,2)); % create a matrix the size of 'k_std x size(A,1)' as H
% 
%         tic
%         [V{mcnt},U{mcnt}]=weakorthonmf(target_R,rand(size(target_R,1),total_topic),rand(total_topic,size(target_R,2)),total_topic,1e-8)
%         speed{mcnt} = toc;


      %% LDA (4th method)

%         mcnt = mcnt + 1; mname{mcnt} = 'LDA'
% 
%         T=total_topic;
%         % Set the hyperparameters
%         BETA=0.01;
%         ALPHA=50/T;
%         % The number of iterations
%         N = 1000;
%         % The random seed
%         SEED = 3;
%         % What output to show (0 = no output; 1 = iterations; 2 = all output)
%         OUTPUT = 1;
% 
%         [ii,jj,ss] = find(R);
% 
%         ss_num = ceil(sum(ss));
% 
%         ii_new = zeros(ss_num,1);
%         jj_new = zeros(ss_num,1);
% 
%         cnt = 1;
%         for i=1:length(ss)
%             ii_new(cnt:(cnt+ss(i)-1)) = ii(i);
%             jj_new(cnt:(cnt+ss(i)-1)) = jj(i);
%             cnt = cnt + ss(i);
%         end
% 
%         tic
%         [V{mcnt}, U{mcnt}, ~ ] = GibbsSamplerLDA( ii_new , jj_new , T , N , ALPHA , BETA , SEED , OUTPUT );
%         speed{mcnt} = toc;
% 
%         U{mcnt} = U{mcnt}';

      %% BoostCF (5th method)

      % =========================================================
        num_beta = 5;
        beta_list = linspace(0, 0.8, num_beta);
        dim_list = 1:3;
        alpha_list = [0 0.3 0.5 0.8];
        
        
        stage = 1000;
      % =========================================================
        
        for ind_dim = 1:length(dim_list)
            for ind_beta = 1:num_beta
                for ind_alpha = 1:length(alpha_list)

                % ===================================================
                    beta = beta_list(ind_beta);
                    dim = dim_list(ind_dim);
                    alpha = alpha_list(ind_alpha);
                    mcnt = mcnt + 1;

                    param.exitAtDeltaPercentage = 1e-4;
                    param.alpha = alpha;
                    param.beta = beta;
                    param.total = stage;
                    param.dim = dim;
                    param.isWithSample = false;
                    param.maxiter = 100;
                    param.max_iter = 100;
                    param.fid = fid;
                    param.hasSocial = true;

                    param.display = 0;
                    param.cf = 'ed';
                    param.conv_eps = 1e-4;
                    param.sparsity = beta;
                
                % ===================================================
                
                    mname{mcnt} = sprintf('BoostCF, beta=%.2f, dim=%d, alpha=%.1f', beta, dim, alpha);
                    fprintf('BoostCF #[%d], beta=%.2f, dim=%d, alpha=%.2f\n',mcnt, beta, dim, alpha);
                    fprintf(fid, 'BoostCF #[%d], beta=%.2f, dim=%d, alpha=%.2f\n',mcnt, beta, dim, alpha);


                    tic;
                    [Ws_wgt, Hs_wgt] = boostCF(target_R, param);
                    speed{mcnt} = toc;

                    V{mcnt} = []; U{mcnt} = [];
                    for i=1:length(Ws_wgt)
                        V{mcnt} = [V{mcnt} Ws_wgt{i}];
                        U{mcnt} = [U{mcnt}; Hs_wgt{i}];
                    end
                end
            end
        end
        
        %% BoostCF (5th method) 2 nd Setting

%         mcnt = mcnt + 1;
%         mname{mcnt} = sprintf('BoostCF')
%         
%         dim = 4;   % number of topics per stage in L-EnsNMF
%         stage = 4; % number of stages in L-EnsNMF
% %         total_topic = dim * stage; % number of total topics
%         beta = 0.6;
% 
%         tic
%         param.beta = beta;
%         param.total = stage;
%         param.dim = dim;
%         param.isWithSample = false;
%         param.maxiter = 100;
%         param.max_iter = 100;
%         
%         param.display = 0;
%         param.cf = 'ed';
%         param.conv_eps = 1e-4;
%         param.sparsity = beta;
%         
% 
%         [Ws_wgt, Hs_wgt, As] = boostCF(target_R, param);
%         speed{mcnt} = toc;
% 
%         V{mcnt} = []; U{mcnt} = [];
%         for i=1:length(Ws_wgt)
%             V{mcnt} = [V{mcnt} Ws_wgt{i}];
%             U{mcnt} = [U{mcnt}; Hs_wgt{i}];
%         end    

      %%

%         Wtopk = {}; Htopk = {}; DocTopk = {}; Wtopk_idx = {};
%         topic_num = 1;
%         for i=1:mcnt
%             [Wtopk{i},Htopk{i},DocTopk{i},Wtopk_idx{i}] = parsenmf(V{i},U{i},dict_new,topk);
% %             mname{i}
% %             Wtopk{i}
%         end                        

        
      % ============= Topic Coherence (in PMI) ============= 
% 
%         % create a zero matrix to store PMI values
%         pmi_vals = zeros(size(Wtopk_idx{1},2),mcnt);  
%         epsilon = 1e-3    % default value
% 
%         for i=1:mcnt
%             for topic_idx=1:size(Wtopk_idx{i},2)
%                 pmi_vals(topic_idx,i) = compute_pmi_log2(R, Wtopk_idx{i}(:,topic_idx),epsilon);
%             end
%         end    
%         
%         pmi_vals

      % =============  Total Document Coverage  ============= 

%         min_nterm_list = 3:10; % min number of keywords doc MUST contain (c2 in the paper)
% 
%         qualtopic = {}; totcvrg= {};
%         qualtopic_mat = zeros(length(min_nterm_list), mcnt);
%         totcvrg_mat = zeros(length(min_nterm_list), mcnt);
% 
%         % totcvrg_mat is a calculation of how many documents k topics covered
%         for min_nterm = min_nterm_list(:)'
%             for idx=1:mcnt
%                 [qualtopic{idx}, totcvrg{idx}] = compute_total_doc_cvrg(R, 	Wtopk_idx{idx}, min_nterm);
%             end
%             qualtopic_mat(min_nterm,:) = mean(cell2mat(qualtopic')');
%             totcvrg_mat(min_nterm,:) = cell2mat(totcvrg);
%         end
%         
%         totcvrg_mat
        
       %% Performance
        mcnt = length(V);
        K_list = 1:30;
        for i = 1:mcnt
            final_result{i} = evaluation(V{i}, U{i}, test_matrix, K_list);
        end
        
%         mcnt = 1:4;
        topN = 7;
        [TopK_result, TopKindex] = select_good_result(final_result, topN, mcnt);
        
        fields = fieldnames(TopK_result{1});
        for i = 1 : length(fields)
            figure;
            hold on;
            for idx = 1:topN
                y = zeros(length(K_list),1);
                for k = 1:length(K_list)
                    y(k) = TopK_result{idx}(k).(fields{i});
                end
                plot(K_list, y); 
            end
            title(fields(i));
            legend(mname(TopKindex));
            hold off;
        end
        
        
        
        
        %% Save Data
        
        saveLog(fid, final_result, K_list, mname);
        fclose(fid);
        
        saveName = saveResult(fidpath);
        save(saveName);
        

        end        

end    

toc