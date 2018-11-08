

% 
% R = [4 5;6 6;8 16];
% w = [1 3;2 4;3 5];
% 
% 
% h0 = rand(2,2);
% 
% hh = fmin_adam(@(h)MSEGradient(R, w, h, 1), h0(:), .2);
% 
% hh = reshape(hh,2,2)




% nDataSetSize = 3;
% vfInput = rand(1, nDataSetSize);
% phiTrue = [3 2];
% fhProblem = @(phi, vfInput) vfInput .* phi(1) + phi(2);
% vfResp = fhProblem(phiTrue, vfInput);% + randn(1, nDataSetSize) * .1;
% plot(vfInput, vfResp, '.'); hold;
% 
% phi0 = [3;20];
% phiHat = fmin_adam(@(phi)LinearRegressionMSEGradients(phi, vfInput, vfResp), phi0, .1)
% plot(vfInput, fhProblem(phiHat, vfInput), '.');
% hold off;


% for idx = evaluate_methods
% %     idx = 1;
%     field = fieldnames(final_result{idx});
%     dim = size(final_result{idx}, 1);
%     K = size(final_result{idx}, 2);
% 
%     for k = 1:length(field)
%         for i = 1:K
%             for j =1:dim
%                 eval(['ranking', '_',field{k},'{', num2str(idx),'}','(', num2str(j) ,',', num2str(i) ,')', ' = ' num2str( final_result{idx}(j,i).(field{k})),';']);
%             end
%         end
%     end
% end

%%

RMSE_min_list = zeros(mcnt,1);
MAE_min_list = zeros(mcnt,1);
for i = 1:mcnt
    RMSE_min_list(i) = min(RMSE{i});
    MAE_min_list(i) = min(MAE{i});
    ranking_recall_max_list(i) = max(max(ranking_recall{i}));
    ranking_arhr_max_list(i) = max(max(ranking_arhr{i}));
    ranking_hr_max_list(i) = max(max(ranking_hr{i}));
    ranking_map_max_list(i) = max(max(ranking_map{i}));
    ranking_NDCG_max_list(i) = max(max(ranking_NDCG{i}));
    ranking_precision_max_list(i) = max(max(ranking_precision{i}));
    
end
%

figure;
plot(RMSE_min_list);
figure;
plot(MAE_min_list);
figure;
plot(ranking_recall_max_list);
title('ranking_recall_max_list');
figure;
plot(ranking_arhr_max_list);
title('ranking_arhr_max_list');
figure;
plot(ranking_hr_max_list);
title('ranking_hr_max_list');
figure;
plot(ranking_map_max_list);
title('ranking_map_max_list');
figure;
plot(ranking_NDCG_max_list);
title('ranking_NDCG_max_list');
figure;
plot(ranking_precision_max_list);
title('ranking_precision_max_list');
