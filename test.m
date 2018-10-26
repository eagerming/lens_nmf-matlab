u = [1 3 5 7 9; 2 4 6 8 10];
v =  [1 2 3 4 5 6; 6 5 4 3 2 1]';
R = v * u;
rng('default'); rng(0); 

R(1,:) = -R(1,:);
R(:,4) = -R(:,4);
R_ori = R;

ind = randperm(30);
R(ind(1:10)) = 0;


% R = [-2.2 2; 0 3; 5 0]
dim = 2;
stage = 100;



param.exitAtDeltaPercentage = 1e-4;

param.total = stage;
param.dim = dim;
param.isWithSample =  false;
param.max_iter = 100;
param.lambda = 0;
param.lambda_social = 0;
param.lambda_item = param.lambda_social;
param.display=0;

param.similarity_threshold = 0.5;

param.learning_rate = 0;
param.is_zero_mask_of_missing = true;


[Ws_wgt, Hs_wgt] = boostCF(R, param);

V = []; U = [];
for i=1:length(Ws_wgt)
    V = [V Ws_wgt{i}];
    U = [U; Hs_wgt{i}];
end

a = V(:,:) * U(:,:) - R;
a(R==0) = 0;



RMSE = zeros(size(V,2),1);
for i = 1:length(RMSE)
    RMSE(i) = sqrt(sum(sum((V(:,1:i) * U(1:i,:) - R_ori).^2)));
end

RMSE