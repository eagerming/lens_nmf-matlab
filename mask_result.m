function WH_masked = mask_result(mask, W, H)
    num_row = size(mask,1);
    num_col = size(mask,2);
    
    [ii, jj] = find(mask);
    
    values = sum(W(ii, :)' .* H(:, jj), 1);
    WH_masked = sparse(ii, jj, values, num_row, num_col);
    
%     WH_masked = sparse(num_row, num_col);
%     for k = 1:length(ii)
%         WH_masked(ii(k),jj(k)) = W(ii(k), :) * H(:, jj(k));
%     end
    
end

% 
% function WH_vec = mask_result(mask, W, H)
%     num_row = size(mask,1);
%     num_col = size(mask,2);
%     
%     ind = find(mask);
%     S = sparse(1:numel(ind), ind, 1,numel(ind), numel(mask));
%     
%     
% 	E=speye(10000);
%     K=kron(E,W);
%     G = S * K;
%     WH_vec = G * H(:);
% 
% end
% 

