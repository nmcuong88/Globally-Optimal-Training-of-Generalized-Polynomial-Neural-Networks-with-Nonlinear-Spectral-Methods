%======================================================================
% Training Generalized Polynomial Nets by our Nonlinear Spectral Method
% Objective: Loss + regul*<1, F> where F is output layer
%
% Written by: Quynh Nguyen
% Last update: 17.05.2016
%======================================================================
function model = train_NLSM(nnet, data, algOptions)

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
model.trainAcc = []; model.trainScore = []; model.trainLoss = [];
model.testAcc = []; model.testScore = []; model.testLoss = [];
%======================================================================
% initialization
for l = 2:nnet.nLayers
    model.W{l} = rand(nnet.layers{l}.nUnits, nnet.layers{l-1}.nUnits);
    model.W{l}(~nnet.layers{l}.mask) = 0;
    model.W{l} = normalize(model.W{l}, nnet.layers{l}.pNorm, nnet.layers{l}.rho, nnet.layers{l}.normType);
    assert(allPos(model.W{l}(nnet.layers{l}.mask)), ['W', num2str(l), ' must be initialized to be positive']);
end
model.initW = model.W;
model.A{1} = data.X; % pass input to 1st layer

%======================================================================
% training phase
success = true; iter = 0; stopCrit = 1;

% NOTE: one can use a higher precison for stopping criteria below
while success && iter < 100 && stopCrit > 1e-7 
    % monitor test performance
    [score, loss, acc] = getScore(nnet, model, algOptions, data, 'test');
    model.testScore = [model.testScore; score];
    model.testLoss = [model.testLoss; loss];
    model.testAcc = [model.testAcc; acc];
    
    iter = iter + 1;
    if algOptions.debug
        disp('===========================================');
        disp(['iter = ', num2str(iter)]);
    end
    
    oldW = model.W;
    
    % forward
    for l = 2:nnet.nLayers
        model.Z{l} = model.W{l} * model.A{l-1};
        model.A{l} = bsxfun(@power, model.Z{l}, nnet.layers{l}.alpha);
        if ~isfield(nnet.layers{l}, 'normFact')
            nnet.layers{l}.normFact = sum(model.A{l}(:));
            if algOptions.debug
                disp(['setting normalization factor to layer ', num2str(l), ': ', num2str(nnet.layers{l}.normFact)]);
            end
        end
        model.A{l} = model.A{l} / nnet.layers{l}.normFact;
    end
    
    [~, ind] = max(model.A{nnet.nLayers});
    model.trainAcc = [model.trainAcc; sum(ind == data.T)/N*100];
    
    % backward
    if strcmpi(algOptions.loss, 'linear')
        B = model.A{nnet.nLayers};
        loss = sum(B(logical(data.Y)))/N;
        regularizer = algOptions.regul * sum(sum(bsxfun(@times, model.A{nnet.nLayers}, nnet.Vec)))/N;
        model.trainScore = [model.trainScore; loss + regularizer];
        model.trainLoss = [model.trainLoss; loss];
        
        model.Delta{nnet.nLayers} = data.Y .* bsxfun(@times, bsxfun(@power, model.Z{nnet.nLayers}, nnet.layers{nnet.nLayers}.alpha-1), nnet.layers{nnet.nLayers}.alpha);
    elseif strcmpi(algOptions.loss, 'logistic')
        B = exp( bsxfun(@minus, model.A{nnet.nLayers}, max(model.A{nnet.nLayers})) );
        B = bsxfun(@rdivide, B, sum(B));
        loss = sum(log(B(logical(data.Y))))/N;
        regularizer = algOptions.regul*sum(sum(bsxfun(@times, model.A{nnet.nLayers}, nnet.Vec)))/N;
        model.trainScore = [model.trainScore; loss + regularizer];
        model.trainLoss = [model.trainLoss; loss];
        
        model.Delta{nnet.nLayers} = (data.Y-B) .* bsxfun(@times, bsxfun(@power, model.Z{nnet.nLayers}, nnet.layers{nnet.nLayers}.alpha-1), nnet.layers{nnet.nLayers}.alpha);
    end
    
    % back-propagate derivatives
    model.Delta{nnet.nLayers} =  model.Delta{nnet.nLayers} + ...
        algOptions.regul * bsxfun(@times, bsxfun(@times, bsxfun(@power, model.Z{nnet.nLayers}, nnet.layers{nnet.nLayers}.alpha-1), nnet.layers{nnet.nLayers}.alpha), nnet.Vec);
    model.Delta{nnet.nLayers} = model.Delta{nnet.nLayers} / nnet.layers{nnet.nLayers}.normFact;
    for l = nnet.nLayers-1:-1:2
        model.Delta{l} = (model.W{l+1}' * model.Delta{l+1}) .* bsxfun(@times, bsxfun(@power, model.Z{l}, nnet.layers{l}.alpha-1), nnet.layers{l}.alpha);
        model.Delta{l} = model.Delta{l} / nnet.layers{l}.normFact;
    end
    
    % compute gradients
    Wgrad = cell(nnet.nLayers, 1);
    for l = 2:nnet.nLayers
        Wgrad{l} = (model.Delta{l} * model.A{l-1}') ./ N;
        Wgrad{l}(~nnet.layers{l}.mask) = 0;
        if min(min(Wgrad{l}(nnet.layers{l}.mask))) < 0
            Wgrad{l}
            Wgrad{l}(nnet.layers{l}.mask)
            disp(['Wgrad', num2str(l), ' must be nonnegative: ', min(min(Wgrad{l}(nnet.layers{l}.mask)))]);
            success = false; pause(5); break;
        end
    end
    if ~success; break; end
    
    % update weights 
    for l = 2:nnet.nLayers
        model.W{l} = (abs(Wgrad{l}).^(1/(nnet.layers{l}.pNorm-1))) .* sign(Wgrad{l});
        model.W{l} = normalize(model.W{l}, nnet.layers{l}.pNorm, nnet.layers{l}.rho, nnet.layers{l}.normType);
    end
    
    % update stopping criteria
    stopCrit = max(arrayfun(@(l) norm(model.W{l}(nnet.layers{l}.mask)-oldW{l}(nnet.layers{l}.mask), inf)/norm(model.W{l}(nnet.layers{l}.mask), inf), 2:nnet.nLayers));
    if algOptions.debug
        disp(['score = ', num2str(model.trainScore(end)), ', acc = ', num2str(model.trainAcc(end))]);
        disp(['StopCrit: ', num2str(stopCrit, 16)]);
    end
end % end while

% evaluate final model
[score, loss, acc] = getScore(nnet, model, algOptions, data, 'train');
model.trainScore = [model.trainScore; score];
model.trainLoss = [model.trainLoss; loss];
model.trainAcc = [model.trainAcc; acc];

[score, loss, acc] = getScore(nnet, model, algOptions, data, 'test');
model.testScore = [model.testScore; score];
model.testLoss = [model.testLoss; loss];
model.testAcc = [model.testAcc; acc];

    % lp-norm sphere normalization
    function D = normalize(C, p, rho, normType)
        if normType == 0
            D = rho * C ./ norm(C(:), p);
        elseif normType == 1
            D = rho * bsxfun(@rdivide, C, sum(C.^p, 2).^(1/p));
        elseif normType == 2
            D = rho * bsxfun(@rdivide, C, sum(C.^p, 1).^(1/p));
        end
    end

    % objective score on training/test data
    function [score, loss, acc] = getScore(nnet, model, algOptions, data, type)
        if strcmpi(type, 'train')
            X = data.X; Y = data.Y; T = data.T;
        elseif strcmpi(type, 'test')
            if ~(isfield(data, 'X_test') && isfield(data, 'Y_test') && isfield(data, 'T_test'))
                score = 0; loss = 0; acc = 0;
                return;
            end
            X = data.X_test; Y = data.Y_test; T = data.T_test;
        end
        A = X;
        for u = 2:nnet.nLayers
            Z = abs(model.W{u} * A);
            A = bsxfun(@power, Z, nnet.layers{u}.alpha);
            if isfield(nnet.layers{u}, 'normFact')
                A = A / nnet.layers{u}.normFact;
            end
        end
        [~, ind] = max(A);
        acc = sum(ind == T)/size(X, 2)*100;
        if strcmpi(algOptions.loss, 'linear')
            loss = sum(A(logical(Y)))/size(X, 2);
            score = loss + algOptions.regul*sum(sum(bsxfun(@times, A, nnet.Vec)))/size(X, 2);
        elseif strcmpi(algOptions.loss, 'logistic')
            BB = exp( bsxfun(@minus, A, max(A)) );
            BB = bsxfun(@rdivide, BB, sum(BB));
            loss = sum(log(BB(logical(Y))))/size(X, 2);
            score = loss + algOptions.regul*sum(sum(bsxfun(@times, A, nnet.Vec)))/size(X, 2);
        end
    end

end % end main func
