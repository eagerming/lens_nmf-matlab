function [MAE, RMSE] = Rating_evaluate(RR, test_matrix, mask_test)
    num = sum(mask_test(:)>0);
    
    RMSE = sqrt( sum(sum((mask_test .* (RR - test_matrix)) .^ 2)) / num);
    MAE = sum(sum((mask_test .* abs(RR - test_matrix)))) / num;
end