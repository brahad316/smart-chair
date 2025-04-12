% minimal_test_network.m
% This is a stand-alone script to test just the network training
% without any optimization

% Load your data
data = readtable('train_25_normalized_3label.csv');
X = data{:, 1:1024};  % [27664×1024] as rows = samples, columns = features
Y = data.three_label;

% Convert Y to categorical
if iscell(Y)
    Y = categorical(Y);
end

% Print sizes
fprintf('Original data:\n');
fprintf('X size: [%d, %d]\n', size(X, 1), size(X, 2));
fprintf('Y size: [%d, %d]\n', size(Y, 1), size(Y, 2));

% Transpose X to [features × samples] format
X_transposed = X';  % Now [1024×27664]

% Create a very simple test network
layers = [
    featureInputLayer(1024, 'Name', 'input')
    fullyConnectedLayer(100, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(2, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];

options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 0.001, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Here's the critical part: work directly with the whole dataset
% but make sure Y is properly formatted
fprintf('Starting network training...\n');
fprintf('X_transposed: [%d, %d]\n', size(X_transposed, 1), size(X_transposed, 2));
fprintf('Y: [%d, %d]\n', size(Y, 1), size(Y, 2));

% Try with the whole dataset
net = trainNetwork(X_transposed, Y, layers, options);

% Save the trained network
save('test_trained_net.mat', 'net');
fprintf('Network saved successfully!\n');