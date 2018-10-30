function [fMSE, vfGrad] = MSEGradient(R, w, h, is_mask)
    
    h = reshape(h,size(w,2), size(R,2));

    if is_mask
        lambda = mask_result(R, w, h);
    else
        lambda = w * h;
    end

    dph = w' * lambda;
    dmh = w' * R;
    
    vfDiff = lambda - R;
    fMSE = 0.5 * mean(vfDiff(:).^2);
    
    vfGrad = dmh - dph;
    vfGrad = -vfGrad(:);
end



function WH_masked = mask_result(A, W, H)
    num_row = size(A,1);
    num_col = size(A,2);
    
    WH_masked = sparse(num_row, num_col);
    [ii, jj] = find(A);
    
    for k = 1:length(ii)
        WH_masked(ii(k),jj(k)) = W(ii(k), :) * H(:, jj(k));
    end
end