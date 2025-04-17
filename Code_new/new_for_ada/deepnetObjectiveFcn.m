% deepnetObjectiveFcn()

function [lossVal, trainedNet] = deepnetObjectiveFcn(X, Y, inputSize, numClasses, params)
    fprintf('========== INPUT DIAGNOSTICS ==========\n');
    fprintf('X size: [%d, %d]\n', size(X, 1), size(X, 2));
    fprintf('Y size: [%d, %d]\n', size(Y, 1), size(Y, 2));
    
    if ~iscategorical(Y)
        Y = categorical(Y);
    end
    if size(Y, 2) > 1
        Y = Y(:);
    end
    
    % Split data
    numSamples = size(X, 1);
    trainRatio = 0.8;
    numTrain = round(trainRatio * numSamples);
    indices = randperm(numSamples);
    trainIdx = indices(1:numTrain);
    valIdx = indices(numTrain+1:end);
    Xtrain = X(trainIdx, :);
    Ytrain = Y(trainIdx);
    Xval = X(valIdx, :);
    Yval = Y(valIdx);

    % Class weights
    classLabels = categories(Ytrain);
    classCounts = countcats(Ytrain);
    classWeights = sum(classCounts) ./ (numel(classCounts) .* classCounts);
 
    % Build layers
    layers = [
        featureInputLayer(inputSize, 'Normalization', 'zscore')];

    for i = 1:params.numHiddenLayers
        layers = [layers
            fullyConnectedLayer(params.numNeurons)
            reluLayer];
        if isfield(params, 'dropoutRate') && params.dropoutRate > 0
            layers = [layers dropoutLayer(params.dropoutRate)];
        end
    end

    layers = [layers
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer('Name', 'output', 'Classes', string(classLabels), 'ClassWeights', classWeights)];

    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', params.miniBatchSize, ...
        'InitialLearnRate', params.initialLearnRate, ...
        'L2Regularization', params.L2Regularization, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'none', ...
        'ValidationData', {Xval, Yval}, ...
        'ValidationFrequency', 20, ...
        'ValidationPatience', 5);

    try
        trainedNet = trainNetwork(Xtrain, Ytrain, layers, options);
        preds = classify(trainedNet, Xval);
        lossVal = 1 - mean(preds == Yval);
    catch ME
        fprintf('ERROR: %s\n', ME.message);
        trainedNet = [];
        lossVal = 1.0;
    end
end
