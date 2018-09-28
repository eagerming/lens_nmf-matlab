function visualBasicVector(W,H)
    if size(W,1) ~= 2
        error('The illustration can be only applied on 2d data');
    end
    
    num = size(H,2);
    startPoint = zeros(2,num);
    
    
    iter = size(W,2);
    for i = 1:iter
        newComponent = W(:,i) * H(i,:);
        endPoint = startPoint + newComponent;
        quiver(startPoint(1,:), startPoint(2,:), endPoint(1,:) - startPoint(1,:), endPoint(2,:) - startPoint(2,:),0);
        startPoint = endPoint;
    end
    

end