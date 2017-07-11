% =======================================================================
% This is a demo version of our NLSM.
% We will run our NLSM on cancer/iris dataset with three classes.
% All architectural parameters like n1, alpha, pw, pu, rhow, rhou, rhox
% will be set at random.
% However, we can use grid search/local search with cross-validation
% as done in our experiments to select these parameters.
%
% NOTICE: In this demo code, we do not check the condition on spectral 
% radius. But in our experiments (see main_NLSM.m and localSearch1+2),
% we always check this condition to obtain global optimality guarantee.
% ========================================================================

load nipsdata; 
ds = nipsdata{1}; % load cancer dataset
% ds = nipsdata{2}; % load iris dataset

% regul: regularizer for sum of output units
algOptions = struct('debug', 0, 'loss', 'logistic', 'regul', 1);

% create training set
data.X = [ds.trainData.X; ones(1, size(ds.trainData.X,2))]; % add bias
data.Y = ds.trainData.Y; % KxN matrix of one-hot output encoding
data.T = ds.trainData.T; % 1xN array of grouth-truth labels

% create test set
data.X_test = [ds.evalData.X; ones(1, size(ds.evalData.X,2))]; % add bias
data.Y_test = ds.evalData.Y; 
data.T_test = ds.evalData.T;

[DimX, N] = size(data.X); % DimX: dimension of input, N: number of samples
K = max(data.T); % number of classes

n1 = 2+randi(10); % number of hidden units
alpha = 1+2*rand(n1,1); % powers of hidden units
rhow = rand; % radius of lp-norm sphere of hidden layer
rhou = rand; % radius of lp-norm sphere of output layer
pw = 50+randi(30); % p-sphere of hidden layer
pu = 100+randi(100); % p-sphere of output layer

% build network
% As in our paper, our objective is given as: losgistic_loss + <output_layer, b>
% Below, vector b is used as nnet.Vec
% Also, every layer has a normalization constant, referred to as 'normFact' 
% below which is 1 by default. Such constant is used only in cases of 
% having numerical issues as the output of each layer blows up.
nnet = struct('nLayers', 3, 'Vec', ones(K, 1));
nnet.layers{1} = struct('name', 'input', 'normFact', 1, 'nUnits', DimX);
nnet.layers{2} = struct('name', 'full', 'normFact', 1, 'nUnits', n1, 'normType', 0, 'pNorm', pw, 'rho', rhou, 'alpha', alpha);
nnet.layers{3} = struct('name', 'full', 'normFact', 1, 'nUnits', K, 'normType', 1, 'pNorm', pu, 'rho', rhow);

% Optially, one can also set sparsity pattern for each layer's weights
% For instance, we can set the hidden layer to be sparse as follows.
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

% compute spectral radius
puprime = nnet.layers{2}.pNorm/(nnet.layers{2}.pNorm-1);
nnet.rhox = max(sum(abs(data.X).^puprime) .^ (1/puprime));
[nnet.srad, ~] = Compute_srad(nnet.layers{3}.pNorm, nnet.layers{2}.pNorm, nnet.layers{3}.rho, nnet.layers{2}.rho, nnet.rhox, K, nnet.layers{2}.alpha, nnet.layers{3}.normType, nnet.layers{2}.normType);
disp(['Spectral radius: ', num2str(nnet.srad)]);

% train model
model = train_NLSM(nnet, data, algOptions);

% display
disp(['Training accuracy: ', num2str(model.trainAcc(end))]);
figure; plot(model.trainScore); xlabel('epoch'); ylabel('training score');
figure; plot(model.trainAcc); xlabel('epoch'); ylabel('training accuracy');
% figure; plot(model.testAcc); xlabel('epoch'); ylabel('test accuracy');
