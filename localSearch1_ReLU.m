function [bestModel, bestNnet] = localSearch1_ReLU(ds)
rng('default');
algOptions = struct('debug', 0, 'loss', 'logistic', 'regul', 0, 'algo', 'pm');

% data.X = ds.trainData.X;
data.X = [ds.trainData.X; ones(1, size(ds.trainData.X,2))];
data.Y = ds.trainData.Y;
data.T = ds.trainData.T;
% data.X_test = ds.evalData.X;
data.X_test = [ds.evalData.X; ones(1, size(ds.evalData.X,2))];
data.Y_test = ds.evalData.Y;
data.T_test = ds.evalData.T;
[DimX, ~] = size(data.X);
K = max(data.T);
MAX_ITER = 500;

% use random search in case an initial network is not provided
bestModel = struct('acc_cv', 0);
for n1 = 2:2:20
    for stepsize = 10.^(-6:2)
        for regul = [0, 10.^(-4:4)]
            % build network
            nnet = struct('nLayers', 3, 'Vec', ones(K, 1));
            nnet.layers{1} = struct('name', 'input', 'normFact', 1, 'nUnits', DimX);
            nnet.layers{2} = struct('name', 'sparse', 'normFact', 1, 'nUnits', n1);
            nnet.layers{3} = struct('name', 'full', 'normFact', 1, 'nUnits', K);

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
                model_cv = train_ReLU(nnet, data_cv, algOptions, regul, stepsize, 1,  MAX_ITER);
                acc_cv = acc_cv + model_cv.trainAcc(end);
                disp(['Cross validate ReLU: ', num2str(model_cv.trainAcc(end)), ' ', num2str(model_cv.testAcc(end))]);
            end
            if bestModel.acc_cv < acc_cv/max(ds.cvid)
                bestModel = train_ReLU(nnet, data, algOptions, regul, stepsize, 1, MAX_ITER);
                bestModel.acc_cv = acc_cv/max(ds.cvid);
                bestNnet = nnet;
            end
            disp(['randomSearchReLU: ', num2str(n1), ' ', num2str(stepsize), ' ', num2str(regul), ': ',...
                num2str(bestModel.trainAcc(end)), ' ', num2str(bestModel.testAcc(end))]);
        end
    end
end
