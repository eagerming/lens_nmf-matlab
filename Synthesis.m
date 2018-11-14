
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
num = 100;

mu1 = [0.3 0.8]; 
sigma1 = [0.001,0.0001;0.0001,0.0008] * 2;
cluster1 = mvnrnd(mu1,sigma1,num);

mu2 = [0.9 0.5]; 
sigma2 = [0.001,0.0001;0.0001,0.0012] * 2; 
cluster2 = mvnrnd(mu2,sigma2,num);


mu3 = [0.7 1];
sigma3 = [0.0015,-0.0001;-0.0001,0.001] * 2; 
cluster3 = mvnrnd(mu3,sigma3,num);


R = [cluster1;cluster2;cluster3];
R = R';
A_original = R;

ratio_set0 = 1/3;
set0_ind = sort(datasample(1:numel(R), round(numel(R) * ratio_set0), 'Replace', false));
test_matrix = zeros(size(R));
test_matrix(set0_ind) = R(set0_ind);
R(set0_ind) = 0;

param.mask = (R ~= 0);

%%
num = 10;

mu1 = [-0.3 0.8 ]; 
sigma1 = [0.01,0.0001;0.0001,0.0108] * 2;
cluster1 = mvnrnd(mu1,sigma1,num);

mu2 = [0.6 0.5]; 
sigma2 = [0.015,0.0001;0.0001,0.0172] * 2; 
cluster2 = mvnrnd(mu2,sigma2,num);

mu3 = [0.1 -0.3];
sigma3 = [0.0215,0.0001;0.0001,0.021] * 2; 
cluster3 = mvnrnd(mu3,sigma3,num);

R = [cluster1;cluster2;cluster3];
R = R';
A_original = R;

social_matrix = zeros(num*3);
for j = 1:3
    for i = 1:num
        r = rand();
        if r > 1
            continue;
        end
        k1 = i + (j-1) * num;
        for ii = 1:num
            r = rand();
            if r > 1
                continue;
            end
            k2 = ii + (j-1) * num;
            social_matrix(k1, k2) = 1;
        end
    end
end
has_trust = true;
if has_trust
    social_matrix = (social_matrix + social_matrix')/2;
    index = sum(social_matrix) > 0;
    social_matrix(:,index) = bsxfun(@rdivide, social_matrix(:,index), sum(social_matrix(:,index)));
end



ratio_set0 = 1/3;
set0_ind = sort(datasample(1:numel(R), round(numel(R) * ratio_set0), 'Replace', false));
test_matrix = zeros(size(R));
test_matrix(set0_ind) = R(set0_ind);
R(set0_ind) = 0;

param.mask = (R ~= 0);

%  Visualization
figure
hold on;
% axis([0 1.2 0 1.2]);
axis square;
scatter(cluster1(:,1),cluster1(:,2));
scatter(cluster2(:,1),cluster2(:,2));
scatter(cluster3(:,1),cluster3(:,2));

[k1,k2] = find(social_matrix);
x = [A_original(1,k1) ;A_original(1,k2)];
y = [A_original(2,k1) ;A_original(2,k2)];
line(x,y, 'Color','red','LineStyle','--');

% scatter(R(1,:), R(2,:));

% quiver(zeros(1,param.total),zeros(1,param.total),W(1,:),W(2,:),0);

% visualBasicVector(W,H);
hold off;


%% Normalization
% % l2 normalization
% A_l2norm = bsxfun(@rdivide,A,sqrt(sum(A.^2)));
% % l1 normalization
% A_l1norm = bsxfun(@rdivide,A,sum(A));
% % tf-idf
% A_idf = tfidf2(A);
% % tf-idf & l2 normalization
% A_l2norm_idf = bsxfun(@rdivide,A,sqrt(sum(A_idf.^2)));
% 
% % use l1-norm/l2-norm weighting/tf-idf
% A = A_l2norm;
% % target_A = A_l2norm;
% % target_A = A_idf;
% % target_A = A_l2norm_idf;
% 
% 
% scatter(A(1, 1:num),A(2, 1:num));
% scatter(A(1, num+1:end),A(2, num+1:end));
% hold on;


%% 
dim = 1;   % number of topics per stage in L-EnsNMF
stage = 3; % number of stages in L-EnsNMF
total_topic = dim * stage; % number of total topics
beta = 0.6;

topk = 10; % number of top keywords to be listed in order within a topic (denoted as c1 in experiment section)


% 
%% Standard NMF
[W0,H0] = nmf(R, total_topic);
% quiver(zeros(1,k_std),zeros(1,k_std), W0(1,:),W0(2,:),0);
% 
% %% 2d NMF
% 
% [W1,H1] = nmfsh_comb_rank2(A, rand(size(A,1),k_std), rand(k_std,size(A,2)));
% quiver(zeros(1,k_std),zeros(1,k_std),W1(1,:),W1(2,:),0);
% 

W = W0;
H = H0;


%% NMF + Sparse
param = [-1 beta];
[W1,H1] = nmfsh_comb(R, total_topic, param);

W1_norm = sqrt(sum(W1.^2));
W1 = bsxfun(@rdivide, W1, W1_norm);
H1 = bsxfun(@times, W1_norm', H1);

W = W1;
H = H1;

%% Sparse NMF
clear param
param.r = total_topic;
param.cf = 'ed';
param.sparsity = beta;
param.max_iter = 1000;
% param.display = 1;
[W2, H2] = sparse_nmf(R, param);

W = W2;
H = H2;

%% Ortho NMF

[W3,H3] = weakorthonmf(R,rand(size(R,1),total_topic),rand(total_topic,size(R,2)),total_topic,1e-8);

W = W3;
H = H3;

%% SLOMA
mcnt = 4;
clear param
param.mask = (R ~= 0);

param.sampleThreshold = 1;
param.similarity_threshold = 0.98;
param.isVisual = true;
param.social_matrix = social_matrix;
param.lambda = 0;
param.lambda_social = 0.1;
param.learning_rate = 0.01;

param.dim_sloma = 2;
param.numOfBlock = 6;

figure
hold on;
% axis([0 1.2 0 1.2]);
axis square;
scatter(cluster1(:,1),cluster1(:,2));
scatter(cluster2(:,1),cluster2(:,2));
scatter(cluster3(:,1),cluster3(:,2));
[Ws_wgt, Hs_wgt, A_sloma] = SLOMA(R, param);
hold off;

K_list = [1 2];
[final_result_sloma, MAE_sloma, RMSE_sloma] = evaluation_sloma(A_sloma, test_matrix, K_list);



%% BoostCF
mcnt = 5;
clear param
param.mask = (R ~= 0);
param.total = 10;
param.dim = 1;
param.isWithSample = true;
param.sampleThreshold = 1;
param.similarity_threshold = 0.98;
param.isVisual = true;
param.social_matrix = social_matrix;
param.lambda = 0.0;
param.lambda_social = 0.1;
param.learning_rate = 0.1;



figure
hold on;
% axis([0 1.2 0 1.2]);
axis square;
scatter(cluster1(:,1),cluster1(:,2));
scatter(cluster2(:,1),cluster2(:,2));
scatter(cluster3(:,1),cluster3(:,2));
tic;
[Ws_wgt, Hs_wgt, As] = boostCF(R, param);
toc
hold off;

W4 = []; H4 = [];
for i=1:length(Ws_wgt)
    W4 = [W4 Ws_wgt{i}];
    H4 = [H4; Hs_wgt{i}];
end

W = W4;
H = H4;
V{mcnt} = W;
U{mcnt} = H;

%

mcnt = length(V);
K_list = [1 2];
evaluate_methods = mcnt:mcnt;

%         profile on;
for i = evaluate_methods
    [final_result{i}, MAE{i}, RMSE{i}] = evaluation(V{i}, U{i}, test_matrix, K_list);
    fprintf('Evaluation method[%d] done\n',i);
end
%         profile viewer;

%
for idx = evaluate_methods
%     idx = 1;
    field = fieldnames(final_result{idx});
    dim = size(final_result{idx}, 1);
    K = size(final_result{idx}, 2);

    for k = 1:length(field)
        for i = 1:K
            for j =1:dim
                eval(['ranking', '_',field{k},'{', num2str(idx),'}','(', num2str(j) ,',', num2str(i) ,')', ' = ' num2str( final_result{idx}(j,i).(field{k})),';']);
            end
        end
    end
end







