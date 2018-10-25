


R = [4 5;6 6;8 16];
w = [1 3;2 4;3 5];


h0 = rand(2,2);

hh = fmin_adam(@(h)MSEGradient(R, w, h, 1), h0(:), .2);

hh = reshape(hh,2,2)




% nDataSetSize = 3;
% vfInput = rand(1, nDataSetSize);
% phiTrue = [3 2];
% fhProblem = @(phi, vfInput) vfInput .* phi(1) + phi(2);
% vfResp = fhProblem(phiTrue, vfInput);% + randn(1, nDataSetSize) * .1;
% plot(vfInput, vfResp, '.'); hold;
% 
% phi0 = [3;20];
% phiHat = fmin_adam(@(phi)LinearRegressionMSEGradients(phi, vfInput, vfResp), phi0, .1)
% plot(vfInput, fhProblem(phiHat, vfInput), '.');
% hold off;