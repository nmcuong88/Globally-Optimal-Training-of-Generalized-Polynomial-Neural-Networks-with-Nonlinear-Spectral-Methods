function [bestModel, bestNnet] = localSearch1_NLSM(ds, initModel, initNnet)
rng('default');
algOptions = struct('debug', 0, 'loss', 'logistic', 'regul', 1, 'algo', 'pm');

% dataORG.X = ds.trainData.X;
dataORG.X = [ds.trainData.X; ones(1, size(ds.trainData.X,2))]; 
dataORG.Y = ds.trainData.Y; 
dataORG.T = ds.trainData.T;
% dataORG.X_test = ds.evalData.X; 
dataORG.X_test = [ds.evalData.X; ones(1, size(ds.evalData.X,2))]; 
dataORG.Y_test = ds.evalData.Y; 
dataORG.T_test = ds.evalData.T;
[DimX, ~] = size(dataORG.X);
K = max(dataORG.T);

% use random search by default
if ~exist('initNet', 'var') || ~exist('initModel', 'var') || isempty(initNet) || isempty(initModel)
    bestNnet = struct('srad', 1e20); 
    bestModel = struct('acc_cv', 0);
    for t = 1:100
        data = dataORG;

        n1 = round(15*rand(1))+10;
        n1 = max(n1, max(size(data.X, 1), K));
        alpha = 1+20*rand(1)*rand(n1,1);

        % tight bounds
        rhow = rand; rhou = rand;
        ma = max(alpha);
        psi1 = rhow*norm(rhou.^alpha, 1);
        psi2 = rhow*norm(alpha.*(rhou.^alpha), 1);
%         normTypeU = double(rand(1)>0.5);
        normTypeU = 0;
        if normTypeU == 0
            pw = round( 2 + 4*K*psi1 + 2*(ma+2*psi2) );
            pu = round( 2 + 2*K*(2*psi1+1) + 2*(ma-1+2*psi2) );
        elseif normTypeU == 1
            pw = round( 2 + 4*K*psi1 + 2*n1*(ma+2*psi2) );
            pu = round( 2 + 2*K*(2*psi1+1) + 2*n1*(ma-1+2*psi2) );
        end
        format short; disp([n1, rhow, rhou, pw, pu]);
        
        % build network
        nnet = struct('nLayers', 3, 'Vec', ones(K, 1));
        nnet.layers{1} = struct('name', 'input', 'normFact', 1, 'nUnits', DimX);
        nnet.layers{2} = struct('name', 'sparse', 'normFact', 1, 'nUnits', n1, 'normType', normTypeU, 'pNorm', pu, 'rho', rhou, 'alpha', alpha);
        nnet.layers{3} = struct('name', 'full', 'normFact', 1, 'nUnits', K, 'normType', 1, 'pNorm', pw, 'rho', rhow);
        
        mask = ones(nnet.layers{2}.nUnits, nnet.layers{1}.nUnits);
        check = 0;
        while(~check)
            for k = 1:size(mask, 2)
                check2 = 0;
                while(~check2)
                    mask(:, k) = double(sprand(size(mask,1), 1, 0.9)>0);
                    if(sum(mask(:,k))>0), check2 = 1; end
                end
            end
            if(min(sum(mask))>0), if(min(sum(mask'))>0); check=1; end, end
        end
        nnet.layers{2}.mask = mask;
        nnet.rhox = 1;
        puprime = nnet.layers{2}.pNorm/(nnet.layers{2}.pNorm-1);
        rhox = max(sum(abs(data.X).^puprime) .^ (1/puprime));
        data.X = nnet.rhox * data.X ./ rhox; 
        data.X_test = nnet.rhox * data.X_test ./ rhox;
        
        acc_cv = 0;
        for fold = 1:max(ds.cvid)
            trainID = ds.cvid~=fold;
            testID = ds.cvid==fold;
            data_cv.X = data.X(:, trainID);
            data_cv.Y = data.Y(:, trainID);
            data_cv.T = data.T(trainID);
            data_cv.X_test = data.X(:, testID);
            data_cv.Y_test = data.Y(:, testID);
            data_cv.T_test = data.T(testID);
            model_cv = train_NLSM(nnet, data_cv, algOptions);
            acc_cv = acc_cv + model_cv.trainAcc(end);
        end
        acc_cv = acc_cv/max(ds.cvid);
        model = train_NLSM(nnet, data, algOptions);
        model.acc_cv = acc_cv;
        
        [nnet.srad, ~] = Compute_srad(nnet.layers{3}.pNorm, nnet.layers{2}.pNorm, nnet.layers{3}.rho, nnet.layers{2}.rho, nnet.rhox, K, nnet.layers{2}.alpha, nnet.layers{3}.normType, nnet.layers{2}.normType);
        if bestNnet.srad < 1
            if nnet.srad < 1 && bestModel.acc_cv < model.acc_cv
                bestNnet = nnet; bestModel = model; 
            end
        elseif nnet.srad < 1 || bestModel.acc_cv < model.acc_cv || ((bestModel.acc_cv == model.acc_cv && nnet.srad<bestNnet.srad))
            bestNnet = nnet; bestModel = model;
        end
        disp(['randomSearch1 ',num2str(t), ': trainAcc ',num2str(model.trainAcc(end)), ', testAcc ', num2str(bestModel.testAcc(end)), ', srad ', num2str(nnet.srad), ', bestSrad ', num2str(bestNnet.srad)]);
    end
else
    bestModel = initModel; bestNnet = initNnet; 
end

% build data for optimal model found by random search
data = dataORG;
puprime = bestNnet.layers{2}.pNorm/(bestNnet.layers{2}.pNorm-1);
rhox = max(sum(abs(data.X).^puprime) .^ (1/puprime));
data.X = bestNnet.rhox * data.X ./ rhox;
data.X_test = bestNnet.rhox * data.X_test ./ rhox;

% local search
max_runs = 10; noimprove = max_runs;
while noimprove > 0
    % check alpha
    for t = 1:10
        nnet = bestNnet;
        nnet.layers{2}.alpha = max(1.1, nnet.layers{2}.alpha+rand*rand*randn(size(nnet.layers{2}.alpha)) );
        
        % tight bounds
%         rhow = nnet.layers{3}.rho; 
%         rhou = nnet.layers{2}.rho;
%         alpha = nnet.layers{2}.alpha;
%         ma = max(alpha);
%         psi1 = rhow*norm(rhou.^alpha, 1);
%         psi2 = rhow*norm(alpha.*(rhou.^alpha), 1);
%         if nnet.layers{2}.normType == 0
%             nnet.layers{3}.pNorm = round( 2 + 4*K*psi1 + 2*(ma+2*psi2) );
%             nnet.layers{2}.pNorm = round( 2 + 2*K*(2*psi1+1) + 2*(ma-1+2*psi2) );
%         elseif nnet.layers{2}.normType == 1
%             nnet.layers{3}.pNorm = round( 2 + 4*K*psi1 + 2*n1*(ma+2*psi2) );
%             nnet.layers{2}.pNorm = round( 2 + 2*K*(2*psi1+1) + 2*n1*(ma-1+2*psi2) );
%         end
        
        acc_cv = 0;
        for fold = 1:max(ds.cvid)
            trainID = ds.cvid~=fold;
            testID = ds.cvid==fold;
            data_cv.X = data.X(:, trainID);
            data_cv.Y = data.Y(:, trainID);
            data_cv.T = data.T(trainID);
            data_cv.X_test = data.X(:, testID);
            data_cv.Y_test = data.Y(:, testID);
            data_cv.T_test = data.T(testID);
            model_cv = train_NLSM(nnet, data_cv, algOptions);
            acc_cv = acc_cv + model_cv.trainAcc(end);
        end
        acc_cv = acc_cv/max(ds.cvid);
        model = train_NLSM(nnet, data, algOptions);
        model.acc_cv = acc_cv;
        
        [nnet.srad, ~] = Compute_srad(nnet.layers{3}.pNorm, nnet.layers{2}.pNorm, nnet.layers{3}.rho, nnet.layers{2}.rho, nnet.rhox, K, nnet.layers{2}.alpha, nnet.layers{3}.normType, nnet.layers{2}.normType);
        if bestNnet.srad < 1
            if nnet.srad < 1 && bestModel.acc_cv < model.acc_cv
                bestNnet = nnet; bestModel = model; noimprove = max_runs;
            end
        elseif bestModel.acc_cv < model.acc_cv || ((bestModel.acc_cv == model.acc_cv && nnet.srad<bestNnet.srad))
            bestNnet = nnet; bestModel = model; noimprove = max_runs;
        elseif nnet.srad < 1
            bestNnet = nnet; bestModel = model;
        end
        disp(['localSearch1 ',num2str(t), ': trainAcc ',num2str(model.trainAcc(end)), ', testAcc ', num2str(bestModel.testAcc(end)), ', srad: ', num2str(nnet.srad), ', bestSrad ', num2str(bestNnet.srad)]);
    end
    
    % check rho
    f1 = [0.8,0.9,0.95,0.975,1.025,1.05,1.1];
    for t = 1:numel(f1)
        nnet = bestNnet;
        nnet.layers{2}.rho = bestNnet.layers{2}.rho * f1(t);

        % tight bounds
%         rhow = nnet.layers{3}.rho; 
%         rhou = nnet.layers{2}.rho;
%         alpha = nnet.layers{2}.alpha;
%         ma = max(alpha);
%         psi1 = rhow*norm(rhou.^alpha, 1);
%         psi2 = rhow*norm(alpha.*(rhou.^alpha), 1);
%         if nnet.layers{2}.normType == 0
%             nnet.layers{3}.pNorm = round( 2 + 4*K*psi1 + 2*(ma+2*psi2) );
%             nnet.layers{2}.pNorm = round( 2 + 2*K*(2*psi1+1) + 2*(ma-1+2*psi2) );
%         elseif nnet.layers{2}.normType == 1
%             nnet.layers{3}.pNorm = round( 2 + 4*K*psi1 + 2*n1*(ma+2*psi2) );
%             nnet.layers{2}.pNorm = round( 2 + 2*K*(2*psi1+1) + 2*n1*(ma-1+2*psi2) );
%         end
        
        acc_cv = 0;
        for fold = 1:max(ds.cvid)
            trainID = ds.cvid~=fold;
            testID = ds.cvid==fold;
            data_cv.X = data.X(:, trainID);
            data_cv.Y = data.Y(:, trainID);
            data_cv.T = data.T(trainID);
            data_cv.X_test = data.X(:, testID);
            data_cv.Y_test = data.Y(:, testID);
            data_cv.T_test = data.T(testID);
            model_cv = train_NLSM(nnet, data_cv, algOptions);
            acc_cv = acc_cv + model_cv.trainAcc(end);
        end
        acc_cv = acc_cv/max(ds.cvid);
        model = train_NLSM(nnet, data, algOptions);
        model.acc_cv = acc_cv;
        
        [nnet.srad, ~] = Compute_srad(nnet.layers{3}.pNorm, nnet.layers{2}.pNorm, nnet.layers{3}.rho, nnet.layers{2}.rho, nnet.rhox, K, nnet.layers{2}.alpha, nnet.layers{3}.normType, nnet.layers{2}.normType);
        if bestNnet.srad < 1
            if nnet.srad < 1 && bestModel.acc_cv < model.acc_cv
                bestNnet = nnet; bestModel = model; noimprove = max_runs;
            end
        elseif bestModel.acc_cv < model.acc_cv || ((bestModel.acc_cv == model.acc_cv && nnet.srad<bestNnet.srad))
            bestNnet = nnet; bestModel = model; noimprove = max_runs;
        elseif nnet.srad < 1
            bestNnet = nnet; bestModel = model;
        end
        disp(['localSearch1 ',num2str(t), ': trainAcc ',num2str(model.trainAcc(end)), ', testAcc ', num2str(bestModel.testAcc(end)), ', srad: ', num2str(nnet.srad), ', bestSrad ', num2str(bestNnet.srad)]);
    end
    
    noimprove = noimprove - 1;
end
