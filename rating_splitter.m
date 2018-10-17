function [train, test] = rating_splitter(R, ratio, israndom)
    [ii,jj,ss] = find(R);
    num = length(ss);
    
    num_train = round(num * ratio);
    num_test = num - num_train;
    
    train = sparse(size(R,1),size(R,2));
    test = sparse(size(R,1),size(R,2));
    
    if israndom
        rng('default')
        rng(0);
        index = randperm(num);
    else
        index = 1:num;
    end
    
    for i = 1:num_train
        ind = index(i);
        train(ii(ind), jj(ind)) = ss(ind);
    end
    for j = i+1 : num
        ind = index(j);
        test(ii(ind), jj(ind)) = ss(ind);
    end
    
    fprintf('Number of ratings of original dataset is %d\n',num);
    fprintf('Training data"s size: %d\n',num_train);
    fprintf('Test data"s size: %d\n',num_test);

end