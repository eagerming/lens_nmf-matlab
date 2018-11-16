function [train, test, mask_train, mask_test] = rating_splitter(mask, R, ratio, israndom)
    [ii,jj] = find(mask);
    ss = R(sub2ind(size(R),ii,jj));
    
    num = length(ss);
    
    num_train = round(num * ratio);
    num_test = num - num_train;
	
    if israndom
        rng('default')
        rng(0);
        index = randperm(num);
    else
        index = 1:num;
    end
    
    mask_train = sparse(ii(index(1:num_train)), jj(index(1:num_train)), 1, size(R,1), size(R,2));
    train = sparse(ii(index(1:num_train)), jj(index(1:num_train)), ss(index(1:num_train)), size(R,1), size(R,2));
%     if numel(train) < numel(R)
%         train(size(R,1), size(R,2)) = 0;
%     end
    
    mask_test = sparse(ii(index(num_train + 1:end)), jj(index(num_train + 1:end)), 1, size(R,1), size(R,2));
    test = sparse(ii(index(num_train + 1:end)), jj(index(num_train + 1:end)), ss(index(1:num_train)), size(R,1), size(R,2));
%     if numel(test) < numel(R)
%         test(size(R,1), size(R,2)) = 0;
%     end
    
    fprintf('Number of ratings of original dataset is %d\n',num);
    fprintf('Training data"s size: %d\n',num_train);
    fprintf('Test data"s size: %d\n',num_test);

end