%======================================================================
% Main Program: One-hidden-layer ReLU networks
%======================================================================

load nipsdata;
N = 7;
bestModels = cell(N, 1);
bestNnets = cell(N, 1);
parfor i = 1:N
    [bestModels{i}, bestNnets{i}] = localSearch1_ReLU(nipsdata{i});
end
if 1
    save ReLU1.mat
end
