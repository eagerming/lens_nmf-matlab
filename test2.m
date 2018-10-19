VV = V{3}, UU = U{3} ;

[ii, jj] = find(R);

for k = 1:length(ii)
    [ii(k), jj(k)]
    RR = VV * UU;
    a = R(ii(k), jj(k))
    b = RR(ii(k), jj(k))
    a - b
end
