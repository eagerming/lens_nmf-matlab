function [ranking_result, MAE_list, RMSE_list]= evaluation(V, U, test_matrix, K_list)

    
    MAE_list = zeros(size(V,2),1);
    RMSE_list = zeros(size(V,2),1);
    
    
    num_of_user = size(U,2);
    
    for dim = 1:size(V,2)
    
        VV = V(:, 1:dim);
        UU = U(1:dim, :);
        recommendation = VV * UU;
        [MAE, RMSE] = Rating_evaluate(recommendation,test_matrix);
        
        MAE_list(dim) = MAE;
        RMSE_list(dim) = RMSE;

        
        num_of_effect_recommendation = 0;
        indicate_result = false(1, num_of_user);

        for user = 1:num_of_user
            recommendation_userI = recommendation(:,user);

            groundtruth_userI = test_matrix(test_matrix(:,user) > 0,user);
            groundtruth_index = find(test_matrix(:,user) > 0);
            num_of_groundtruth = length(groundtruth_userI);

            if num_of_groundtruth <= eps
                continue;
            end

            num_of_effect_recommendation = num_of_effect_recommendation + 1;
            [~, index] = sort(groundtruth_userI,'descend');

            lastK = 0;
            for i = 1:length(K_list)
                k = K_list(i);
                if num_of_groundtruth > k 
                    reserved_index = index(1:k);
                else
                    reserved_index = index;
                end
                thisK = length(reserved_index);
                if thisK == lastK
                    result_userI_k = result_userI_k_last;
                else
                    result_userI_k = evaluation_per_user(... 
                    recommendation_userI, num_of_groundtruth, ...
                    groundtruth_userI(reserved_index), ...
                    groundtruth_index(reserved_index));
                end
                lastK = thisK;
                result_userI_k_last = result_userI_k;

                indicate_result(user) = true;
                all_result(i,user) = result_userI_k;

            end
        end

        field = fieldnames(all_result);

        for user = 1:num_of_user
            if ~ indicate_result(user)
                continue;
            end

            for k = 1:length(K_list)
                for i = 1:length(field)
                    last.(field{i}) = 0;
                end

                for i = 1:length(field)
                    ranking_result(dim,k).(field{i}) = last.(field{i}) + all_result(k,user).(field{i}) / num_of_effect_recommendation;
                    last.(field{i}) = ranking_result(dim,k).(field{i});
                end

            end
        end
    end
    
end