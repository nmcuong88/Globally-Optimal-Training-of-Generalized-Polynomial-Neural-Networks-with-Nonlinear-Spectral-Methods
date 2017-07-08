%======================================================================
% Training ReLU Nets by Batch-SGD
% Objective: Loss + 1/2*regul*||W||^2
% 
% Written by: Quynh Nguyen
% Last update: 17.05.2016
%======================================================================
function model = train_ReLU(nnet, data, algOptions, regul, stepsize, batchRatio, maxDataPass, initW)
rng('default');
[DimX, N] = size(data.X);

%======================================================================
% computing params
model.A = cell(nnet.nLayers, 1);
model.Z = cell(nnet.nLayers, 1);
model.Delta = cell(nnet.nLayers, 1);
model.W = cell(nnet.nLayers, 1);
model.trainAcc = [];
model.trainScore = [];
model.stepsize = stepsize;
model.regul = regul;
%======================================================================
% initialization
for l = 2:nnet.nLayers
    if ~exist('initW', 'var') || isempty(initW)
        model.W{l} = randn(nnet.layers{l}.nUnits, nnet.layers{l-1}.nUnits);
    else
        model.W{l} = initW{l};
    end
end
[model.trainScore, model.trainAcc] = getScore(nnet, model, algOptions, data);
[model.testScore, model.testAcc] = getTestScore(nnet, model, algOptions, data);

%======================================================================
% training phase
batchSize = max(5, floor(batchRatio*N));
itersPerPass = floor((N-1)/batchSize)+1;
iter = 0;
while iter < itersPerPass*maxDataPass
    iter = iter + 1;
    
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
        model.A{l} = max(0, model.Z{l});
    end
    
    % monitor training accuracy
    if ~mod(iter, itersPerPass)
        [score, acc] = getScore(nnet, model, algOptions, data);
        model.trainScore = [model.trainScore; score];
        model.trainAcc = [model.trainAcc; acc];
        
        [score, acc] = getTestScore(nnet, model, algOptions, data);
        model.testScore = [model.testScore; score];
        model.testAcc = [model.testAcc; acc];
        
        if algOptions.debug
            disp(['SGD-iter ', num2str(iter), ': score ', num2str(model.trainScore(end)), ', acc ', num2str(model.trainAcc(end)), ', stepsize ', num2str(stepsize)]);
        end
    end
    
    % backward
    if strcmpi(algOptions.loss, 'linear')
        B = model.A{nnet.nLayers};
        model.Delta{nnet.nLayers} = data.Y(:, batchID);
    elseif strcmpi(algOptions.loss, 'logistic')
        B = exp( bsxfun(@minus, model.A{nnet.nLayers}, max(model.A{nnet.nLayers})) );
        B = bsxfun(@rdivide, B, sum(B));
        model.Delta{nnet.nLayers} = (data.Y(:, batchID)-B);
    end
    
    for l = nnet.nLayers-1:-1:2
        model.Delta{l} = (model.W{l+1}' * model.Delta{l+1}) .* max(0,sign(model.Z{l}));
    end
    
    % compute gradients
    Wgrad = cell(nnet.nLayers, 1);
    for l = 2:nnet.nLayers
        Wgrad{l} = (model.Delta{l} * model.A{l-1}') ./ numel(batchID);
        Wgrad{l} = Wgrad{l} + model.regul*model.W{l};
    end
        
    for l = 2:nnet.nLayers
%         model.W{l} = model.W{l} + stepsize*Wgrad{l};
        model.W{l} = model.W{l} + stepsize*Wgrad{l}/norm(Wgrad{l}(:));
    end
end % end while

% save the number of required passes through the data
% model.nPasses = floor((iter-1)/itersPerPass)+1;

    % objective score on training set
    function [score, acc] = getScore(nnet, model, algOptions, data)
        A = data.X;
        for u = 2:nnet.nLayers
            Z = model.W{u} * A;
            A = max(0, Z);
        end
        [~, ind] = max(A);
        acc = sum(ind == data.T)/size(data.X, 2)*100;
        s = 0;
        for u = 2:nnet.nLayers
            s = s + norm(model.W{u}(:))^2;
        end
        if strcmpi(algOptions.loss, 'linear')
            score = sum(log(A(logical(data.Y))))/size(data.X, 2) + .5*model.regul*s;
        elseif strcmpi(algOptions.loss, 'logistic')
            B = exp( bsxfun(@minus, A, max(A)) );
            B = bsxfun(@rdivide, B, sum(B));
            score = sum(log(B(logical(data.Y))))/size(data.X, 2) + .5*model.regul*s;
        end
    end

    % objective score on test set
    function [score, acc] = getTestScore(nnet, model, algOptions, data)
        A = data.X_test;
        for u = 2:nnet.nLayers
            Z = model.W{u} * A;
            A = max(0, Z);
        end
        [~, ind] = max(A);
        acc = sum(ind == data.T_test)/size(data.X_test, 2)*100;
        s = 0; 
        for u = 2:nnet.nLayers
            s = s + norm(model.W{u}(:))^2;
        end
        if strcmpi(algOptions.loss, 'linear')
            score = sum(log(A(logical(data.Y_test))))/size(data.X_test, 2) + .5*model.regul*s;
        elseif strcmpi(algOptions.loss, 'logistic')
            B = exp( bsxfun(@minus, A, max(A)) );
            B = bsxfun(@rdivide, B, sum(B));
            score = sum(log(B(logical(data.Y_test))))/size(data.X_test, 2) + .5*model.regul*s;
        end
    end
end % end main func
