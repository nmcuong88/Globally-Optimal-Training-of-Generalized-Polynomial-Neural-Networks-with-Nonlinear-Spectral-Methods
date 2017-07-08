%======================================================================
% SVM with cross validation of hyper-perameters
%======================================================================

function testSVM()
load nipsdata;
svmAcc = zeros(numel(nipsdata), 2);

% using libsvm + cross validate C, gamma
addpath('libsvm-3.21/matlab');
parfor i = 1:numel(nipsdata)
    acc = zeros(1, 2);
    nFolds = nipsdata{i}.nFolds;
    
    % linear kernel: nothing to cross validate
    C = 2.^[-5:20];
    cvAcc = zeros(1, numel(C));
    for k = 1:numel(C)
        s = 0;
        for fold = 1:nFolds
            trainId = nipsdata{i}.cvid ~= fold;
            X = nipsdata{i}.trainData.X(:, trainId)'; 
            T = nipsdata{i}.trainData.T(trainId)';
            X_test = nipsdata{i}.trainData.X(:, ~trainId)'; 
            T_test = nipsdata{i}.trainData.T(~trainId)';
            model = svmtrain(T, X, ['-c ', num2str(C(k)), ' -t 0'] );
            [labels, ~, ~] = svmpredict(T_test, X_test, model);
            s = s + sum(labels==T_test) / numel(T_test) * 100;
        end
        cvAcc(k) = s/nFolds;
        disp([nipsdata{i}.name, ', linear kernel: C=', num2str(C(k)), ', acc=', num2str(cvAcc(k))]);
    end
    [val, k] = max(cvAcc);
    X = nipsdata{i}.trainData.X'; T = nipsdata{i}.trainData.T';
    X_test = nipsdata{i}.evalData.X'; T_test = nipsdata{i}.evalData.T';
    model = svmtrain(T, X, ['-c ', num2str(C(k)), ' -t 0'] );
    [labels, ~, ~] = svmpredict(T_test, X_test, model);
    acc(1) = sum(labels==T_test) / numel(T_test) * 100;    
    disp('===============================================');
    disp([nipsdata{i}.name, ', linear kernel: bestC=', num2str(C(k)), ', bestAcc=', num2str(acc(1))]);
    disp('===============================================');
    
    % RBF kernel: cross validate C, gamma
    [C, gamma] = ndgrid(2.^[-5:20], 2.^[-15:3]);
    C = C(:);
    gamma = gamma(:);
    cvAcc = zeros(1, numel(C));
    for k = 1:numel(C)
        s = 0;
        for fold = 1:nFolds
            trainId = nipsdata{i}.cvid ~= fold;
            X = nipsdata{i}.trainData.X(:, trainId)'; 
            T = nipsdata{i}.trainData.T(trainId)';
            X_test = nipsdata{i}.trainData.X(:, ~trainId)'; 
            T_test = nipsdata{i}.trainData.T(~trainId)';
            model = svmtrain(T, X, ['-c ', num2str(C(k)), ' -g ', num2str(gamma(k))] );
            [labels, ~, ~] = svmpredict(T_test, X_test, model);
            s = s + sum(labels==T_test) / numel(T_test) * 100;
        end
        cvAcc(k) = s/nFolds;
        disp([nipsdata{i}.name, ', RBF kernel: C=', num2str(C(k)), ', gamma=', num2str(gamma(k)), ', acc=', num2str(cvAcc(k))]);
    end
    [val, k] = max(cvAcc);
    X = nipsdata{i}.trainData.X'; T = nipsdata{i}.trainData.T';
    X_test = nipsdata{i}.evalData.X'; T_test = nipsdata{i}.evalData.T';
    model = svmtrain(T, X, ['-c ', num2str(C(k)), ' -g ', num2str(gamma(k))] );
    [labels, ~, ~] = svmpredict(T_test, X_test, model);
    acc(2) = sum(labels==T_test) / numel(T_test) * 100;    
    disp('===============================================');
    disp([nipsdata{i}.name, ', RBF kernel: bestC=', num2str(C(k)), ', bestGamma=', num2str(gamma(k)), ', bestAcc=', num2str(acc(2))]);
    disp('===============================================');
    
    svmAcc(i, :) = acc;
end
save(['svmAcc.mat'], 'svmAcc');
