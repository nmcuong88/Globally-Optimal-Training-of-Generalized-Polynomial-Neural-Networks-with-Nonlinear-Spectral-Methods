%======================================================================
% Main Program: Nonlinear Spectral Method
%======================================================================
load nipsdata;
N = 7;
tab = zeros(N, 2);
bestModels = cell(N, 2);
bestNnets = cell(N, 2);
for i = 1:N
    % 1-hidden layer
    [bestModels{i, 1}, bestNnets{i, 1}] = localSearch1_NLSM(nipsdata{i});
    tab(i, 1) = bestModels{i, 1}.testAcc(end);
    
    % 2-hidden layer
    [bestModels{i, 2}, bestNnets{i, 2}] = localSearch2_NLSM(nipsdata{i});
    tab(i, 2) = bestModels{i, 2}.testAcc(end);
end
if 1
    save NLSM.mat
end
