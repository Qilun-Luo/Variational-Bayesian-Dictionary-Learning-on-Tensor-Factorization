%% 2dmatrix hard_threshold(2dmatrix)
% S - sparse 2dmatrix
% O - outliers 2dmatrix
%
function O = hard_threshold(S)
  
%   % beta = 0.5*(3*std(S(:))/20)^2; % min beta, lower bound: suppose SNR <= 20
%   beta = 0.5*(std(S(:)))^2; % begin beta, start from a big value
%   
%   % direct hard thresholding if no smoothness
%   O = double(0.5*S.^2 > beta);
  
    beta2 = (std(S(:)));
    O2 = double(abs(S) > beta2);

    beta = (std(S(:)))^2;
    O = double(S.^2 > beta);
    
%     E = O-O2;
%     norm(E(:))
end
