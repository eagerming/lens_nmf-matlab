VV = V{2};
UU = U{2};

[MAE, RMSE] = Rating_evaluate(VV,UU,test_matrix)

C = zeros(size(VV,2),1);
for kk = 1:size(VV,2)
    RR = VV(:,1:kk) * UU(1:kk,:);


    %% Training
    % [ii, jj] = find(R);
    % for k = 1:length(ii)
    %     [ii(k), jj(k)]
    %     a = R(ii(k), jj(k))
    %     b = RR(ii(k), jj(k))
    %     a - b
    % end


    %% Test


    [ii, jj] = find(test_matrix);
    c = 0;
    for k = 1:length(ii)
        [ii(k), jj(k)];
        a = test_matrix(ii(k), jj(k));
        b = RR(ii(k), jj(k));
        c = c + (a - b)^2;
    end
    C(kk) = sqrt(c);
end
C
%% 






