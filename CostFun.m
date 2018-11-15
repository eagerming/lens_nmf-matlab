function [cost, derivative] = CostFun(W,H,R,mask, isH, costparam)
    
    dim = numel(W)/size(mask,1);
    W = reshape(W, size(mask,1), dim);
    H = reshape(H, dim, size(mask,2));
    WH = mask_result(mask, W, H);
    
    
    lambda_social = costparam.lambda_social;
    lambda_item = costparam.lambda_item;
    lambda = costparam.lambda;
    if lambda_social > 0
        social_matrix = costparam.social_matrix;
    end
    if lambda_item > 0
        item_matrix = costparam.item_matrix;
    end
    
    div = sum(sum((R - WH) .^ 2))/2;
    cost = div;
    if lambda > 0
        cost = cost + lambda * (sum(sum(W .^2 ))/2 + sum(sum(H .^2 ))/2);
    end
    if lambda_item > 0
        cost = cost + lambda_item * sum(sum((W - (W' * item_matrix)').^2));
    end
    if lambda_social > 0
        cost = cost + lambda_social * sum(sum((H - H * social_matrix).^2));
    end
    
    
    if isH % update H
        dph = W' * WH;
        dmh = W' * R;
        if lambda
            dph = dph + lambda * H;
        end
        if lambda_social > 0
            dph = dph + lambda_social * H;
            dmh = dmh + lambda_social * (H * social_matrix);
        end
        derivative = dph - dmh;
        derivative = derivative(:);
    else
%         dpw = WH * H' + bsxfun(@times, sum((R * H' + lambda_item * (W' * item_matrix)') .* W), W) + (lambda_item + lambda) * W;
%         dmw = R * H' + lambda_item * (W' * item_matrix)' + bsxfun(@times, sum((WH * H' + (lambda_item + lambda) * W) .* W), W);

        temp1 = R * H';
        temp2 = WH * H';

        dpw = WH * H';
        dmw = R * H';
        
        if lambda
            dpw = dpw + lambda * W; 
%             dmw = dmw + bsxfun(@times, sum((lambda * W) .* W), W);
            temp2 = temp2 + lambda * W;
        end
        
        if lambda_item
            temp1 = temp1 + lambda_item * (W' * item_matrix)';
            temp2 = temp2 + lambda_item * W;
            
            dpw = dpw + lambda_item * W;
            dmw = dmw + lambda_item * (W' * item_matrix)';
        end
        
        dpw = dpw + bsxfun(@times, sum(temp1 .* W), W);
        dmw = dmw + bsxfun(@times, sum(temp2 .* W), W);
        
        derivative = dpw - dmw;
        derivative = derivative(:);
    end
end




