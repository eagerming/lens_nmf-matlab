u = [1 3 5 7 9; 2 4 6 8 10];
v =  [1 2 3 4 5 6; 6 5 4 3 2 1]';
R = v * u;
rng('default'); rng(0); 

R(1,:) = -R(1,:);
R(:,4) = -R(:,4);
R_ori = R;

ind = randperm(30);
R_testData = zeros(size(R));
R_testData(ind(1:10)) = R(ind(1:10));
R(ind(1:10)) = 0;



% R = [-2.2 2; 0 3; 5 0]
dim = 1;
stage = 10;



param.exitAtDeltaPercentage = 1e-4;

param.total = stage;
param.dim = dim;
param.isWithSample =  false;
param.max_iter = 100;
param.lambda = 0;
param.lambda_social = 0;
param.lambda_item = param.lambda_social;
param.display = 0;

param.similarity_threshold = 0;

param.learning_rate = 0;
param.is_mask = true;


[Ws_wgt, Hs_wgt] = boostCF(R, param);

V = []; U = [];
for i=1:length(Ws_wgt)
    V = [V Ws_wgt{i}];
    U = [U; Hs_wgt{i}];
end

a = V(:,:) * U(:,:) - R;
a(R==0) = 0;



%% Random (1st method)
VV{1} = rand(size(R,1), 5);
UU{1} = rand(5, size(R,2));

VV{2} = V;
UU{2} = U;


K_list = [1 3 6];
        
evaluate_methods = [1 2];

% profile on;
for i = evaluate_methods
    [final_result_test{i}, MAE_test{i}, RMSE_test{i}] = evaluation(VV{i}, UU{i}, R_testData, K_list);
end
% profile viewer;


for idx = evaluate_methods
%     idx = 1;
    field = fieldnames(final_result_test{idx});
    dim = size(final_result_test{idx}, 1);
    K = size(final_result_test{idx}, 2);

    for k = 1:length(field)
        for i = 1:K
            for j =1:dim
                eval(['ranking_test', '_',field{k},'{', num2str(idx),'}','(', num2str(j) ,',', num2str(i) ,')', ' = ' num2str( final_result_test{idx}(j,i).(field{k})),';']);
            end
        end
    end
end

% 
% RMSE = zeros(size(V,2),1);
% for i = 1:length(RMSE)
%     RMSE(i) = sqrt(sum(sum((V(:,1:i) * U(1:i,:) - R_ori).^2)));
% end
% RMSE