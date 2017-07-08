%======================================================================
% Main Program: Two-hidden-layer ReLU networks
%======================================================================

load nipsdata;
N = 7;
bestModels = cell(N, 1);
bestNnets = cell(N, 1);
parfor i = 1:N
    [bestModels{i}, bestNnets{i}] = localSearch2_ReLU(nipsdata{i});
end
if 1
    save ReLU2.mat
end
