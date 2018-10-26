VV = V{49};
UU = U{49};

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






