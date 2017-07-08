%======================================================================
% Preparing datasets for experiments
%======================================================================

allPos = @(x)(all(x(:) > 0));
allNonneg = @(x)(all(x(:) >= 0));
allNeg = @(x)(all(x(:) < 0));
allNonpos = @(x)(all(x(:) <= 0));
noNanInf = @(x)(sum(isnan(x(:)))==0&&sum(isinf(x(:)))==0);

nipsdata = {};
ds = {};
% UCI datasets
ds{end+1} = loadDatasetByName('cancer_dataset');
ds{end+1} = loadDatasetByName('iris_dataset');
ds{end+1} = loadDatasetByName('banknote');
ds{end+1} = loadDatasetByName('blood');
ds{end+1} = loadDatasetByName('haberman');
ds{end+1} = loadDatasetByName('seeds');
ds{end+1} = loadDatasetByName('pima');
% ds{end+1} = loadDatasetByName('simpleclass_dataset');

for i = 1:numel(ds)
    data = ds{i};
    [DimX, N] = size(data.X);
    if ~isfield(data, 'X_test')
        X = data.X;
        T = data.T;
        nClasses = max(T);
        
        id = crossvalind('Kfold', T, 5);
        trainId = id~=1;
        
        trainData.X = X(:, trainId);
        trainData.T = T(trainId);
        N = size(trainData.X, 2);
        trainData.Y = full(sparse(trainData.T, 1:N, ones(1, N), nClasses, N));
        
        evalData.X = X(:, ~trainId);
        evalData.T = T(~trainId);
        N = size(evalData.X, 2);
        evalData.Y = full(sparse(evalData.T, 1:N, ones(1, N), nClasses, N));
    else
        trainData = struct('X', data.X, 'Y', data.Y, 'T', data.T);
        evalData = struct('X', data.X_test, 'Y', data.Y_test, 'T', data.T_test);
    end
    nFolds = 5;
    cvid = crossvalind('Kfold', trainData.T, nFolds);
    
    nipsdata{end+1} = struct('trainData', trainData, 'evalData', evalData, ...
        'nFolds', nFolds, 'cvid', cvid, 'name', ds{i}.name);
    nipsdata{end}
end

% check data partitions
for i = 1:numel(nipsdata)
    C1 = unique(nipsdata{i}.trainData.T);
    C2 = unique(nipsdata{i}.evalData.T);
    assert(numel(C1)==numel(C2)&&numel(C1)==max([C1(:);C2(:)]), [num2str(i), '. ', nipsdata{i}.name, ': train/test partition: ', num2str(C1), ' || ', num2str(C2)]);
    disp([nipsdata{i}.name, ', #distribution of classes in trainData: ', num2str(hist(nipsdata{i}.trainData.T, C1))]);
    disp([nipsdata{i}.name, ', #distribution of classes in evalData: ', num2str(hist(nipsdata{i}.evalData.T, C2))]);
    for k = 1:nipsdata{i}.nFolds
        trainId = nipsdata{i}.cvid ~= k;
        C = unique(nipsdata{i}.trainData.T(trainId));
        assert(numel(C)==max([C1(:);C2(:)]), [num2str(i), '. ', nipsdata{i}.name, ': data partition does not contain all classes']);
        a = hist(nipsdata{i}.trainData.T(trainId), C);
        disp([nipsdata{i}.name, ', fold ', num2str(k), ': ', num2str(a)]);
    end
end
nipsdata_copy = nipsdata;

% ============================================================
% SAVE DATA

% ============================================================
% Translating samples to positive orthant as required by our
% Nonlinear Spectral Method. The same is done on test data.
nipsdata = nipsdata_copy;
for i = 1:numel(nipsdata)
    mi = min(nipsdata{i}.trainData.X(:));
    ma = max(nipsdata{i}.trainData.X(:));
    nipsdata{i}.trainData.X = (nipsdata{i}.trainData.X - mi) / ma + 1e-10;
    nipsdata{i}.evalData.X = (nipsdata{i}.evalData.X - mi) / ma + 1e-10;
    assert(allPos(nipsdata{i}.trainData.X), 'data are not positive');
    assert(noNanInf(nipsdata{i}.trainData.X) && noNanInf(nipsdata{i}.evalData.X), 'data contain nan/inf');
end
save('nipsdata.mat', 'nipsdata');

% ============================================================
% Create another dataset with zero-mean and unit-variance
nipsdata = nipsdata_copy;
for i = 1:numel(nipsdata)
    X = nipsdata{i}.trainData.X;
    X = bsxfun(@minus, X, mean(X, 2));
    a = std(X,0,2); id=a==0; a(id)=1;
    X = bsxfun(@rdivide, X, a); 
    X(id, :) = [];
    
    mi = min(X(:));
    ma = max(X(:));
    %X = bsxfun(@minus, X, min(X,[],2)) + 1e-10;
    X = (X-mi)/ma + 1e-10;
    nipsdata{i}.trainData.X = X;
    
    X = nipsdata{i}.evalData.X;
    X = bsxfun(@minus, X, mean(X, 2));
    a = std(X,0,2); a(id)=1;
    X = bsxfun(@rdivide, X, a); 
    X(id, :) = [];
    X = (X-mi)/ma + 1e-10;
    nipsdata{i}.evalData.X = X;
    
    assert(allPos(nipsdata{i}.trainData.X), 'data are not positive');
    assert(noNanInf(nipsdata{i}.trainData.X) && noNanInf(nipsdata{i}.evalData.X), 'data contain nan/inf');
end
save('nipsdata_norm1.mat', 'nipsdata');

