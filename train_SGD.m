%======================================================================
% Training Generalized Polynomial Nets using Batch-SGD
% Objective: Loss + regul*<1,F>, where F is output layer
% 
% Written by: Quynh Nguyen
% Last update: 17.05.2016
%======================================================================
function model = trainNNet_SGD(nnet, data, algOptions, stepsize, batchRatio, maxDataPass, initW)
rng('default');

% network settings
[DimX, N] = size(data.X);
for l = 2:nnet.nLayers
    if ~isfield(nnet.layers{l}, 'mask')
        nnet.layers{l}.mask = ones(nnet.layers{l}.nUnits, nnet.layers{l-1}.nUnits);
    end
    nnet.layers{l}.mask = logical(nnet.layers{l}.mask);
    
    if ~isfield(nnet.layers{l}, 'alpha')
        nnet.layers{l}.alpha = ones(nnet.layers{l}.nUnits, 1);
    end
    nnet.layers{l}.alpha = reshape(nnet.layers{l}.alpha, numel(nnet.layers{l}.alpha), 1);
end

% utilities
allZero = @(x)(all(x(:)==0));
allPos = @(x)(all(x(:) > 0));
allNonneg = @(x)(all(x(:) >= 0));
allNeg = @(x)(all(x(:) < 0));
allNonpos = @(x)(all(x(:) <= 0));

%======================================================================
% computing params
model.A = cell(nnet.nLayers, 1);
model.Z = cell(nnet.nLayers, 1);
model.Delta = cell(nnet.nLayers, 1);
model.W = cell(nnet.nLayers, 1);
model.trainAcc = []; model.trainScore = [];
model.testAcc = []; model.testScore = [];

%======================================================================
% initialization
for l = 2:nnet.nLayers
    if ~exist('initW', 'var') || isempty(initW)
        % model.W{l} = rand(nnet.layers{l}.nUnits, nnet.layers{l-1}.nUnits);
        model.W{l} = ones(nnet.layers{l}.nUnits, nnet.layers{l-1}.nUnits);
    else
        model.W{l} = initW{l};
    end
    model.W{l}(~nnet.layers{l}.mask) = 0;
    model.W{l} = normalize(model.W{l}, nnet.layers{l}.pNorm, nnet.layers{l}.rho, nnet.layers{l}.normType);
    assert(allPos(model.W{l}(nnet.layers{l}.mask)), ['W', num2str(l), ' must be initialized to be positive']);
end

%======================================================================
% training phase
batchSize = max(5, floor(batchRatio*N));
itersPerPass = floor((N-1)/batchSize)+1;
iter = 0;
while iter < itersPerPass*maxDataPass
    % monitor training accuracy
    if ~mod(iter, itersPerPass)
        [score, acc] = getScore(nnet, model, algOptions, data);
        model.trainScore = [model.trainScore; score];
        model.trainAcc = [model.trainAcc; acc];
        
        [score, acc] = getTestScore(nnet, model, algOptions, data);
        model.testScore = [model.testScore; score];
        model.testAcc = [model.testAcc; acc];
        if algOptions.debug
            disp(['SGD-iter ', num2str(iter), '/', num2str(itersPerPass*maxDataPass), ': score ', num2str(model.trainScore(end)), ', acc ', num2str(model.trainAcc(end)), ', stepsize ', num2str(stepsize)]);
        end
    end
    
    iter = iter + 1;
    t0 = tic;
    
    % re-shuffle the data at the beginning of each epoch
    if ~mod(iter-1, itersPerPass)
        perm = randperm(N);
        data.X = data.X(:, perm);
        data.Y = data.Y(:, perm);
        data.T = data.T(perm);
    end
    
    % pass input to 1st layer
    part = mod(iter-1, itersPerPass)+1;
    batchID = (part-1)*batchSize+1 : min(part*batchSize, N);
    model.A{1} = data.X(:, batchID);
    
    % forward
    for l = 2:nnet.nLayers
        model.Z{l} = model.W{l} * model.A{l-1};
        model.A{l} = bsxfun(@power, model.Z{l}, nnet.layers{l}.alpha);
    end
    
    % backward
    if strcmpi(algOptions.loss, 'linear')
        B = model.A{nnet.nLayers};
        model.Delta{nnet.nLayers} = data.Y(:, batchID) .* bsxfun(@times, bsxfun(@power, model.Z{nnet.nLayers}, nnet.layers{nnet.nLayers}.alpha-1), nnet.layers{nnet.nLayers}.alpha);
    elseif strcmpi(algOptions.loss, 'logistic')
        B = exp( bsxfun(@minus, model.A{nnet.nLayers}, max(model.A{nnet.nLayers})) );
        B = bsxfun(@rdivide, B, sum(B));
        model.Delta{nnet.nLayers} = (data.Y(:, batchID)-B) .* bsxfun(@times, bsxfun(@power, model.Z{nnet.nLayers}, nnet.layers{nnet.nLayers}.alpha-1), nnet.layers{nnet.nLayers}.alpha);
    end
    % add the derivative of this regularizer: regul*<1, F> 
    model.Delta{nnet.nLayers} =  model.Delta{nnet.nLayers} + ...
        algOptions.regul * bsxfun(@times, bsxfun(@times, bsxfun(@power, model.Z{nnet.nLayers}, nnet.layers{nnet.nLayers}.alpha-1), nnet.layers{nnet.nLayers}.alpha), nnet.Vec);
    % back-propagate derivatives
    for l = nnet.nLayers-1:-1:2
        model.Delta{l} = (model.W{l+1}' * model.Delta{l+1}) .* bsxfun(@times, bsxfun(@power, model.Z{l}, nnet.layers{l}.alpha-1), nnet.layers{l}.alpha);
    end
    
    % compute gradients
    Wgrad = cell(nnet.nLayers, 1);
    for l = 2:nnet.nLayers
        Wgrad{l} = (model.Delta{l} * model.A{l-1}') ./ numel(batchID);
        Wgrad{l}(~nnet.layers{l}.mask) = 0;
    end
    
    % compute gradient norm
    s = 0;
    for l = 2:nnet.nLayers
        s = s + norm(Wgrad{l}(:))^2;
    end
    s = sqrt(s);
    disp(['gradient norm of all weights: ', num2str(s)]);
    
    % SGD update
    for l = 2:nnet.nLayers
        % normalize the gradient if necessary
        model.W{l} = model.W{l} + stepsize*Wgrad{l}/s;
        
        % lp-norm sphere normalization
        model.W{l} = cvxProjection(model.W{l}, nnet.layers{l}.mask, nnet.layers{l}.pNorm, nnet.layers{l}.rho, nnet.layers{l}.normType);
    end
    disp(['SGD-projection takes ', num2str(toc(t1))]);
    disp(['SGD-iter takes ', num2str(toc(t0))]);
end % end while

% save the number of required passes through the data
model.nPasses = floor((iter-1)/itersPerPass)+1;

% evaluate final model
[score, acc] = getScore(nnet, model, algOptions, data);
model.trainScore = [model.trainScore; score];
model.trainAcc = [model.trainAcc; acc];

[score, acc] = getTestScore(nnet, model, algOptions, data);
model.testScore = [model.testScore; score];
model.testAcc = [model.testAcc; acc];

    % lp-norm sphere normalization
    function D = normalize(C, p, rho, normType)
        if normType == 0
            D = rho * C ./ norm(C(:), p);
        elseif normType == 1
            D = rho * bsxfun(@rdivide, C, sum(abs(C).^p, 2).^(1/p));
        elseif normType == 2
            D = rho * bsxfun(@rdivide, C, sum(abs(C).^p, 1).^(1/p));
        end
    end

    % objective score over training set
    function [score, acc] = getScore(nnet, model, algOptions, data)
        A = data.X;
        for u = 2:nnet.nLayers
            Z = model.W{u} * A;
            A = bsxfun(@power, Z, nnet.layers{u}.alpha);
        end
        [~, ind] = max(A);
        acc = sum(ind == data.T)/size(data.X, 2)*100;
        if strcmpi(algOptions.loss, 'linear')
            score = sum(A(logical(data.Y)))/size(data.X, 2) + algOptions.regul*sum(sum(bsxfun(@times, A, nnet.Vec)))/size(data.X, 2);
        elseif strcmpi(algOptions.loss, 'logistic')
            B = exp( bsxfun(@minus, A, max(A)) );
            B = bsxfun(@rdivide, B, sum(B));
            score = sum(log(B(logical(data.Y))))/size(data.X, 2) + algOptions.regul*sum(sum(bsxfun(@times, A, nnet.Vec)))/size(data.X, 2);
        end
    end

    % objective score over test set
    function [score, acc] = getTestScore(nnet, model, algOptions, data)
        A = data.X_test;
        for u = 2:nnet.nLayers
            Z = model.W{u} * A;
            A = bsxfun(@power, Z, nnet.layers{u}.alpha);
        end
        [~, ind] = max(A);
        acc = sum(ind == data.T_test)/size(data.X_test, 2)*100;
        if strcmpi(algOptions.loss, 'linear')
            score = sum(A(logical(data.Y_test)))/size(data.X_test, 2) + algOptions.regul*sum(sum(bsxfun(@times, A, nnet.Vec)))/size(data.X_test, 2);
        elseif strcmpi(algOptions.loss, 'logistic')
            B = exp( bsxfun(@minus, A, max(A)) );
            B = bsxfun(@rdivide, B, sum(B));
            score = sum(log(B(logical(data.Y_test))))/size(data.X_test, 2) + algOptions.regul*sum(sum(bsxfun(@times, A, nnet.Vec)))/size(data.X_test, 2);
        end
    end

end % end main func
