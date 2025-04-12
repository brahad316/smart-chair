function [lossVal, trainedNet] = deepnetObjectiveFcn(X, Y, inputSize, numClasses, params)
    % Add extensive debugging
    fprintf('========== INPUT DIAGNOSTICS ==========\n');
    fprintf('X size: [%d, %d]\n', size(X, 1), size(X, 2));
    fprintf('Y size: [%d, %d]\n', size(Y, 1), size(Y, 2));
    fprintf('Y class: %s\n', class(Y));
    fprintf('inputSize: %d\n', inputSize);
    fprintf('numClasses: %d\n', numClasses);
    
    % Ensure dimensions are correct - X should be [samples x features]
    if size(X, 2) ~= inputSize
        fprintf('ERROR: X dimensions do not match inputSize\n');
        lossVal = 1.0;
        trainedNet = [];
        return;
    end
    
    % Ensure Y is categorical and column vector
    if ~iscategorical(Y)
        fprintf('Converting Y to categorical\n');
        Y = categorical(Y);
    end
    
    % Make sure Y is a column vector
    if size(Y, 2) > 1
        fprintf('Reshaping Y to column vector\n');
        Y = Y(:);
    end
    
    fprintf('After preprocessing:\n');
    fprintf('X size: [%d, %d]\n', size(X, 1), size(X, 2));
    fprintf('Y size: [%d, %d]\n', size(Y, 1), size(Y, 2));
    
    % Split data manually to ensure dimensions are correct
    fprintf('Splitting data...\n');
    numSamples = size(X, 1);
    trainRatio = 0.8;
    numTrain = round(trainRatio * numSamples);
    
    % Create random indices for splitting
    indices = randperm(numSamples);
    trainIdx = indices(1:numTrain);
    valIdx = indices(numTrain+1:end);
    
    % Create training and validation sets
    Xtrain = X(trainIdx, :);  % [samples x features]
    Ytrain = Y(trainIdx);
    
    Xval = X(valIdx, :);  % [samples x features]
    Yval = Y(valIdx);
    
    fprintf('Training set dimensions:\n');
    fprintf('Xtrain: [%d, %d] (samples x features)\n', size(Xtrain, 1), size(Xtrain, 2));
    fprintf('Ytrain: [%d, %d]\n', size(Ytrain, 1), size(Ytrain, 2));
    fprintf('First 5 elements of Ytrain: ');
    disp(Ytrain(1:min(5, length(Ytrain))));
    
    fprintf('Validation set dimensions:\n');
    fprintf('Xval: [%d, %d] (samples x features)\n', size(Xval, 1), size(Xval, 2));
    fprintf('Yval: [%d, %d]\n', size(Yval, 1), size(Yval, 2));
    
    % Define layers
    layers = [
        featureInputLayer(inputSize, 'Normalization', 'zscore')];
    
    for i = 1:params.numHiddenLayers
        layers = [layers
            fullyConnectedLayer(params.numNeurons)
            reluLayer];
    end
    
    layers = [layers
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
    
    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', params.initialLearnRate, ...
        'L2Regularization', params.L2Regularization, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'none', ...
        'ValidationData', {Xval, Yval}, ...
        'ValidationFrequency', 20, ...
        'ValidationPatience', 5);
    
    % Before training, check dimensions one last time
    fprintf('===== FINAL CHECK BEFORE TRAINING =====\n');
    fprintf('Xtrain size: [%d, %d]\n', size(Xtrain, 1), size(Xtrain, 2));
    fprintf('Ytrain size: [%d, %d]\n', size(Ytrain, 1), size(Ytrain, 2));
    
    try
        fprintf('Attempting to train network...\n');
        trainedNet = trainNetwork(Xtrain, Ytrain, layers, options);
        fprintf('Training successful!\n');
        
        % Evaluate on validation set
        preds = classify(trainedNet, Xval);
        lossVal = 1 - mean(preds == Yval);
        fprintf('Validation loss: %.4f\n', lossVal);
    catch ME
        fprintf('ERROR: %s\n', ME.message);
        fprintf('Training failed, returning high loss\n');
        trainedNet = [];
        lossVal = 1.0;
    end
end