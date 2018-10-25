function [fMSE, vfGrad] = LinearRegressionMSEGradients(phi, vfInput, vfResp)
   % - Compute mean-squared error using the current parameter estimate
   vfRespHat = vfInput .* phi(1) + phi(2);
   vfDiff = vfRespHat - vfResp;
   fMSE = mean(vfDiff.^2) / 2;
   
   % - Compute the gradient of MSE for each parameter
   vfGrad(1) = mean(vfDiff .* vfInput);
   vfGrad(2) = mean(vfDiff);
   vfGrad = vfGrad;
end