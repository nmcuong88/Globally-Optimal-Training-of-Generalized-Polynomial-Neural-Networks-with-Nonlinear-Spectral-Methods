%======================================================================
% Written by: Quynh Nguyen
% Last update: 17.05.2016
%======================================================================

function data = loadDatasetByName(s)
data = {};
data.name = s;
allNonneg = @(x)(all(x(:) >= 0));
if ~isempty(regexp(s, '\w*.mat'))
    load(['datasets/', s]);
    data.X = XD';
    data.T = YD';
    data.Y = full(sparse(data.T, 1:size(data.X,2), ones(1, size(data.X,2)), 2, size(data.X,2)));
    data.name = strrep(s, '.mat', '');
elseif strcmpi(s, 'simpleclass_dataset') || strcmpi(s, 'cancer_dataset') || strcmpi(s, 'iris_dataset') 
    [data.X, data.Y] = eval(s);
    [data.T, ~] = find(data.Y==1);
    data.T = reshape(data.T, 1, numel(data.T));
    data.X = bsxfun(@minus, data.X, min(data.X, [], 2));
elseif strcmpi(s, 'banknote')
    A = dlmread('datasets/uci/data_banknote_authentication.txt');
    N = size(A, 1); nClasses = 2;
    data.X = A(:,1:end-1)';
    data.T = A(:, end)' + 1;
    data.Y = full(sparse(data.T, 1:N, ones(1, N), nClasses, N));
elseif strcmpi(s, 'blood')
    A = dlmread('datasets/uci/transfusion.txt');
    N = size(A, 1); nClasses = 2;
    data.X = A(:,1:end-1)';
    data.T = A(:, end)' + 1;
    data.Y = full(sparse(data.T, 1:N, ones(1, N), nClasses, N));
elseif strcmpi(s, 'iris')
    fid = fopen('datasets/uci/bezdekIris.txt', 'rt');
    C = textscan(fid, '%f,%f,%f,%f,%s');
    data.X = [C{1}, C{2}, C{3}, C{4}]';
    N = size(data.X, 2); nClasses = 3;
    data.T = [ones(1, 50), 2*ones(1, 50), 3*ones(1, 50)];
    data.Y = full(sparse(data.T, 1:N, ones(1, N), nClasses, N));
elseif strcmpi(s, 'haberman')
    A = dlmread('datasets/uci/haberman.txt');
    N = size(A, 1); nClasses = 2;
    data.X = A(:,1:end-1)';
    data.T = A(:, end)';
    data.Y = full(sparse(data.T, 1:N, ones(1, N), nClasses, N));
elseif strcmpi(s, 'seeds')
    A = dlmread('datasets/uci/seeds_dataset.txt');
    N = size(A, 1); nClasses = 3;
    data.X = A(:,1:end-1)';
    data.T = A(:, end)';
    data.Y = full(sparse(data.T, 1:N, ones(1, N), nClasses, N));
elseif strcmpi(s, 'pima')
    A = dlmread('datasets/uci/pima-indians-diabetes.txt');
    N = size(A, 1); nClasses = 2;
    data.X = A(:,1:end-1)';
    data.T = A(:, end)' + 1;
    data.Y = full(sparse(data.T, 1:N, ones(1, N), nClasses, N));
end
assert(numel(unique(data.T))==max(data.T), ['error loading ', num2str(data.name), ': wrong class indices']);
