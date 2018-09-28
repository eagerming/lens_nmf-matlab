function mat = gauss2d(mat, sigma, center)
gsize = size(mat);
for r=1:gsize(1)
    for c=1:gsize(2)
        mat(r,c) = gaussC(r,c, sigma, center);
    end
end
function val = gaussC(x, y, sigma, center)
xc = center(1);
yc = center(2);
exponent = ((x-xc).^2 + (y-yc).^2)./(2*sigma);
val       = (exp(-exponent));  