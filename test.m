u = [1 3 5 7 9; 2 4 6 8 10];
v =  [1 2 3 4 5 6; 6 5 4 3 2 1]';
R = v* u;
rng('default'); rng(0); 

ind = randperm(30);

R(ind(1:10)) = 0;

beta = 0
dim = 1
alpha = 0.5
param.hasSocial = false;


param.exitAtDeltaPercentage = 1e-4;
param.alpha = alpha;
param.beta = beta;
param.total = stage;
param.dim = dim;
param.isWithSample =  false;
param.maxiter = 100;
param.max_iter = 100;
param.fid = fid;

param.display = 1;
param.cf = 'ed';
param.conv_eps = 1e-4;
param.sparsity = beta;


[Ws_wgt, Hs_wgt] = boostCF(R, param);

V = []; U = [];
for i=1:length(Ws_wgt)
    V = [V Ws_wgt{i}];
    U = [U; Hs_wgt{i}];
end

a = V * U - R;
a(R==0) = 0