function [train, test] = rating_splitter(R, ratio, israndom)
    [ii,jj,ss] = find(R);
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
    
    train = sparse(ii(index(1:num_train)), jj(index(1:num_train)), ss(index(1:num_train)));
    if numel(train) < numel(R)
        train(size(R,1), size(R,2)) = 0;
    end
    
    test = sparse(ii(index(num_train + 1:end)), jj(index(num_train + 1:end)), ss(index(num_train + 1:end)));
    if numel(test) < numel(R)
        test(size(R,1), size(R,2)) = 0;
    end
    
    fprintf('Number of ratings of original dataset is %d\n',num);
    fprintf('Training data"s size: %d\n',num_train);
    fprintf('Test data"s size: %d\n',num_test);

end