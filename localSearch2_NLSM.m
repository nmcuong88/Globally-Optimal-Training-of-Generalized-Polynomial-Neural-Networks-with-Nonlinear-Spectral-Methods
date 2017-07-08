function [bestModel, bestNnet] = localSearch2_NLSM(ds, initModel, initNnet)
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

% use random search in case an initial network is not provided
if ~exist('initNet', 'var') || ~exist('initModel', 'var') || isempty(initNet) || isempty(initModel)
    bestNnet = struct('srad', 1e20); 
    bestModel = struct('acc_cv', 0);
    for t = 1:150
        data = dataORG;
        
        param = randi(3);
        if param==1; tu = 0; else; tu = 1; end;
        if param==2; tv = 0; else; tv = 1; end;
        
        if strcmpi(ds.name, 'cancer_dataset')
            n2 = 1+randi(5); n1 = 1+randi(5);
            alpha = 1+2*rand(n1, 1); beta = 1+2*rand(n2, 1);
            rhou=rand*rand; rhov=rand*rand; rhow=rand*rand; 
        elseif strcmpi(ds.name, 'blood')
            n2 = 1+randi(5); n1 = 1+randi(5);
            alpha = 1+2*rand(n1, 1); beta = 1+2*rand(n2, 1);
            rhou=rand*rand; rhov=1; rhow=rand;
        elseif strcmpi(ds.name, 'haberman')
            n2 = 1+randi(10); n1 = 1+randi(10);
            alpha = 1+3*rand(n1, 1); beta = 1+3*rand(n2, 1);
            rhou=rand*rand; rhov=1; rhow=rand;
        elseif strcmpi(ds.name, 'seeds')
            n2 = 1+randi(5); n1 = 1+randi(5);
            alpha = 1+3*rand(n1, 1); beta = 1+3*rand(n2, 1);
            rhou=rand; rhov=rand; rhow=rand;
        elseif strcmpi(ds.name, 'pima')
            n2 = 1+randi(10);
            n1 = 1+randi(10);
            alpha = 1+3*rand(n1, 1);
            beta = 1+3*rand(n2, 1);
            rhou=rand*rand; rhov=rand*rand; rhow=rand;
        else
            n2 = 1+randi(5); n1 = 1+randi(5);
            alpha = 1+2*rand(n1, 1); beta = 1+2*rand(n2, 1);
            rhou=rand*rand; rhov=rand*rand; rhow=rand;
        end
                
        ma = max(alpha);
        mb = max(beta);
        mia = min(alpha);
        psi = rhou.^(mia/1e4)*n1^(1-mia/1e4);
        thetaW = rhow*norm((rhov*psi).^beta, 1);
        thetaV = rhow*norm(beta.*((rhov*psi).^beta), 1);
        thetaU = rhow*ma*norm(beta.*((rhov*psi).^beta), 1);
        pu = round( 20 + 2*n2*(2*thetaW+1) + 2*(2*thetaV+mb) + 2*(2*thetaU-2+ma+mb) ) + randi(50);
        pv = round( 20 + 2*n2*(2*thetaW+1) + 2*(2*thetaV-1+mb) + 2*(2*thetaU+mb) ) + randi(50);
        pw = round( 20 + 4*n2*thetaW + 2*(2*thetaV+1) + 2*(2*thetaU+ma) ) + randi(50);
        
        % build network
        nnet = struct('nLayers', 4, 'Vec', ones(K, 1), 'AGparam', param);
        nnet.layers{1} = struct('name', 'input', 'normFact', 1, 'nUnits', DimX);
        nnet.layers{2} = struct('name', 'sparse', 'normFact', 1, 'nUnits', n1, 'normType', tu, 'pNorm', pu, 'rho', rhou, 'alpha', alpha);
        nnet.layers{3} = struct('name', 'sparse', 'normFact', 1, 'nUnits', n2, 'normType', tv, 'pNorm', pv, 'rho', rhov, 'alpha', beta);
        nnet.layers{4} = struct('name', 'full', 'normFact', 1, 'nUnits', K, 'normType', 1, 'pNorm', pw, 'rho', rhow);
        
        for l = 2:nnet.nLayers-1
            mask = ones(nnet.layers{l}.nUnits, nnet.layers{l-1}.nUnits);
            check = 0;
            while(~check)
                for k = 1:size(mask, 2)
                    check2 = 0;
                    while(~check2)
                        mask(:, k) = double(sprand(size(mask,1), 1, 0.95)>0);
                        if(sum(mask(:,k))>0), check2 = 1; end
                    end
                end
                if(min(sum(mask))>0), if(min(sum(mask'))>0); check=1; end, end
            end
            nnet.layers{l}.mask = mask;
        end
        
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
        
        [nnet.srad, ~] = Compute_srad_2(nnet.layers{4}.pNorm, nnet.layers{3}.pNorm, nnet.layers{2}.pNorm, nnet.layers{4}.rho, nnet.layers{3}.rho, nnet.layers{2}.rho, nnet.rhox, nnet.layers{nnet.nLayers}.nUnits, nnet.layers{2}.alpha, nnet.layers{3}.alpha, nnet.AGparam);
        if bestNnet.srad < 1
            if nnet.srad < 1 && bestModel.acc_cv < model.acc_cv
                bestNnet = nnet; bestModel = model;
            end
        elseif nnet.srad < 1 || bestModel.acc_cv < model.acc_cv || ((bestModel.acc_cv == model.acc_cv && nnet.srad<bestNnet.srad))
            bestNnet = nnet; bestModel = model;
        end
        disp(['randomSearch2 ',num2str(t), ': trainAcc ',num2str(model.trainAcc(end)), ', testAcc ', num2str(bestModel.testAcc(end)), ', srad ', num2str(nnet.srad), ', bestSrad ', num2str(bestNnet.srad)]);
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
max_runs = 5; noimprove = max_runs;
while noimprove > 0
    % check alpha
    for t = 1:100
        nnet = bestNnet;
        for l = 2:nnet.nLayers-1
            nnet.layers{l}.alpha = max(1.1, nnet.layers{l}.alpha+rand*rand*rand*randn(size(nnet.layers{l}.alpha)) );
        end
        
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
        
        [nnet.srad, ~] = Compute_srad_2(nnet.layers{4}.pNorm, nnet.layers{3}.pNorm, nnet.layers{2}.pNorm, nnet.layers{4}.rho, nnet.layers{3}.rho, nnet.layers{2}.rho, nnet.rhox, nnet.layers{nnet.nLayers}.nUnits, nnet.layers{2}.alpha, nnet.layers{3}.alpha, nnet.AGparam);
        if bestNnet.srad < 1
            if nnet.srad < 1 && bestModel.acc_cv < model.acc_cv
                bestNnet = nnet; bestModel = model; noimprove = max_runs;
            end
        elseif bestModel.acc_cv < model.acc_cv || ((bestModel.acc_cv == model.acc_cv && nnet.srad<bestNnet.srad))
            bestNnet = nnet; bestModel = model; noimprove = max_runs;
        elseif nnet.srad < 1
            bestNnet = nnet; bestModel = model;
        end
        disp(['localSearch2 ',num2str(t), ': trainAcc ',num2str(model.trainAcc(end)), ', testAcc ', num2str(bestModel.testAcc(end)), ', srad: ', num2str(nnet.srad), ', bestSrad ', num2str(bestNnet.srad)]);
    end
    
    % check rho
    [f1, f2] = ndgrid([0.9,0.95,0.975,1.025,1.05,1.1]);
    f1 = f1(:);
    f2 = f2(:);
    %     f1 = [0.8,0.9,0.95,0.975,1.025,1.05,1.1];
    for t = 1:numel(f1)
        nnet = bestNnet;
        nnet.layers{2}.rho = bestNnet.layers{2}.rho * f1(t);
        nnet.layers{3}.rho = bestNnet.layers{3}.rho * f2(t);
        
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
        
        [nnet.srad, ~] = Compute_srad_2(nnet.layers{4}.pNorm, nnet.layers{3}.pNorm, nnet.layers{2}.pNorm, nnet.layers{4}.rho, nnet.layers{3}.rho, nnet.layers{2}.rho, nnet.rhox, nnet.layers{nnet.nLayers}.nUnits, nnet.layers{2}.alpha, nnet.layers{3}.alpha, nnet.AGparam);
        if bestNnet.srad < 1
            if nnet.srad < 1 && bestModel.acc_cv < model.acc_cv
                bestNnet = nnet; bestModel = model; noimprove = max_runs;
            end
        elseif bestModel.acc_cv < model.acc_cv || ((bestModel.acc_cv == model.acc_cv && nnet.srad<bestNnet.srad))
            bestNnet = nnet; bestModel = model; noimprove = max_runs;
        elseif nnet.srad < 1
            bestNnet = nnet; bestModel = model;
        end
        disp(['localSearch2 ',num2str(t), ': trainAcc ',num2str(model.trainAcc(end)), ', testAcc ', num2str(bestModel.testAcc(end)), ', srad: ', num2str(nnet.srad), ', bestSrad ', num2str(bestNnet.srad)]);
    end
    
    noimprove = noimprove - 1;
end
