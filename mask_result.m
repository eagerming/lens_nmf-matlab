function WH_masked = mask_result(A, W, H)
    num_row = size(A,1);
    num_col = size(A,2);
    
    
    WH_masked = sparse(num_row, num_col);
    [ii, jj] = find(A);
    
    for k = 1:length(ii)
        WH_masked(ii(k),jj(k)) = W(ii(k), :) * H(:, jj(k));
    end
end