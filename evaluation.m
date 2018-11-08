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

        
%         num_of_effect_recommendation = 0;
        effect_k = zeros(length(K_list),1);
%         indicate_result = false(1, num_of_user);

        num_of_groundtruth_user = zeros(num_of_user,1);
        for user = 1:num_of_user
            recommendation_userI = recommendation(:,user);

            groundtruth_userI = test_matrix(test_matrix(:,user) > 0,user);
            groundtruth_index = find(test_matrix(:,user) > 0);
            groundtruth_index_missing = test_matrix(:,user) == 0;
            num_of_groundtruth = length(groundtruth_userI);
            num_of_groundtruth_user(user) = num_of_groundtruth;

            if num_of_groundtruth <= eps
                continue;
            end

%             num_of_effect_recommendation = num_of_effect_recommendation + 1;
            [~, index] = sort(groundtruth_userI,'descend');
            
            recommendation_userI(groundtruth_index_missing) = 0;
            index_of_groundtruth_in_ordered_recommendation = get_index(recommendation_userI, groundtruth_userI(index), groundtruth_index(index));
            
            lastK = 0;
            for i = 1:length(K_list)
                k = K_list(i);
%                 if num_of_groundtruth > k 
%                     reserved_index = index(1:k);
%                 else
%                     reserved_index = index;
%                 end
                if num_of_groundtruth >= k 
                    reserved_index = index_of_groundtruth_in_ordered_recommendation(1:k);
                    effect_k(i) = effect_k(i) + 1;
                else
                    break;
                    reserved_index = index_of_groundtruth_in_ordered_recommendation;
                end
                thisK = length(reserved_index);
                if thisK == lastK
                    result_userI_k = result_userI_k_last;
                else
                    
                    groundtruth = groundtruth_userI(index(1:length(reserved_index)));
                    result_userI_k = Ranking_evaluate(reserved_index, num_of_groundtruth, groundtruth);
                    
%                     result_userI_k = evaluation_per_user(... 
%                     recommendation_userI, num_of_groundtruth, ...
%                     groundtruth_userI(reserved_index), ...
%                     groundtruth_index(reserved_index));
                end
                lastK = thisK;
                result_userI_k_last = result_userI_k;

                all_result(i,user) = result_userI_k;

            end
        end

        field = fieldnames(all_result);

        
        for k = 1:length(K_list)
            for i = 1:length(field)
                last.(field{i}) = 0;
            end
            for user = 1:num_of_user
                if ~ (num_of_groundtruth_user(user) >= K_list(k))
                    continue;
                end
                for i = 1:length(field)
                    ranking_result(dim,k).(field{i}) = last.(field{i}) + all_result(k,user).(field{i}) / effect_k(k);
                    last.(field{i}) = ranking_result(dim,k).(field{i});
                end

            end
        end
    end
    
    
    
    
    
    
    
end


function index_of_groundtruth_in_ordered_recommendation = get_index(recommendation, groundtruth, groundtruth_index)
    K = length(groundtruth);
    
    new_groundtruth_index = adjust_position_of_equal_value_according_to_recommendation(recommendation, groundtruth, groundtruth_index);
    index_groundtruth_in_rec = inf(length(recommendation),1);
    for i = 1:K
        index_groundtruth_in_rec(new_groundtruth_index(i)) = i;
    end
    
    [~, index_bigRec] = sort(recommendation,'descend');
    new_pos =  index_groundtruth_in_rec(index_bigRec);
    [~, index_of_groundtruth_in_ordered_recommendation] = sort(new_pos);
    index_of_groundtruth_in_ordered_recommendation(K+1:end) = [];
end


function new_groundtruth_index = adjust_position_of_equal_value_according_to_recommendation(recommendation, groundtruth, groundtruth_index)
    [~,first_pointer] = unique(groundtruth, 'stable');
    new_groundtruth_index = groundtruth_index;
    
    if length(first_pointer) < length(groundtruth)
        first_pointer(end + 1) = length(groundtruth) + 1;
        num_list = diff(first_pointer);
        first_pointer(end) = [];
        
        for i = 1:length(first_pointer)
            if num_list(i) > 1
                index_sub = groundtruth_index(first_pointer(i) : first_pointer(i) + num_list(i) - 1);
                rec_sub = recommendation(index_sub);
                [~, ind_rec] = sort(rec_sub,'descend');
                adjust_index_sub = index_sub(ind_rec);
                new_groundtruth_index(first_pointer(i) : first_pointer(i) + num_list(i) - 1) = adjust_index_sub;
            end
        end
    end

end