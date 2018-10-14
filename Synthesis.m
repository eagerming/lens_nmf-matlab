
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


A = [cluster1;cluster2;cluster3];
A_original = A;
A = A';

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
[W0,H0] = nmf(A, total_topic);
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
[W1,H1] = nmfsh_comb(A, total_topic, param);

W1_norm = sqrt(sum(W1.^2));
W1 = bsxfun(@rdivide, W1, W1_norm);
H1 = bsxfun(@times, W1_norm', H1);

W = W1;
H = H1;

%% Sparse NMF
param.r = total_topic;
param.cf = 'ed';
param.sparsity = beta;
param.max_iter = 1000;
% param.display = 1;
[W2, H2] = sparse_nmf(A, param);

W = W2;
H = H2;

%% Ortho NMF

[W3,H3] = weakorthonmf(A,rand(size(A,1),total_topic),rand(total_topic,size(A,2)),total_topic,1e-8);

W = W3;
H = H3;


%% BoostCF

param.beta = beta;
param.total = stage;
param.dim = dim;
param.isWithSample = false;

[Ws_wgt, Hs_wgt, As] = boostCF(A, param);
W4 = []; H4 = [];
for i=1:length(Ws_wgt)
    W4 = [W4 Ws_wgt{i}];
    H4 = [H4; Hs_wgt{i}];
end

W = W4;
H = H4;


%% 
figure
hold on;
axis([0 1.2 0 1.2]);
axis square;
scatter(cluster1(:,1),cluster1(:,2));
scatter(cluster2(:,1),cluster2(:,2));
scatter(cluster3(:,1),cluster3(:,2));


quiver(zeros(1,total_topic),zeros(1,total_topic),W(1,:),W(2,:),0);
visualBasicVector(W,H)
hold off;



