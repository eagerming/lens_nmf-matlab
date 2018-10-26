function [MAE, RMSE] = Rating_evaluate(RR,test_matrix)
    [ii, jj] = find(test_matrix);
    
    
    
    %% Training
    % [ii, jj] = find(R);
    % for k = 1:length(ii)
    %     [ii(k), jj(k)]
    %     a = R(ii(k), jj(k))
    %     b = RR(ii(k), jj(k))
    %     a - b
    % end


    %% Test
    RMSE = 0;
    MAE = 0;
    for k = 1:length(ii)
        a = test_matrix(ii(k), jj(k));
        b = RR(ii(k), jj(k));
        MAE = MAE + abs(a-b);
        RMSE = RMSE + (a - b)^2;
    end
    MAE = MAE / length(ii);
    RMSE = sqrt(RMSE/length(ii));
    
    %% 
end