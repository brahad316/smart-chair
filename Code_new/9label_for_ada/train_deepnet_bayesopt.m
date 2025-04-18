% train_deepnet_bayesopt.m - Fixed version

outputDir = 'training_results'; % Folder to save outputs
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% STEP 1: Load data
fprintf('Loading data...\n');
data = readtable('train_25_9label.csv');
X = data{:, 1:1024};
Y = data.posture; % change this if your column is named differently

fprintf('Data loaded:\n');
fprintf('X size: [%d, %d]\n', size(X, 1), size(X, 2));
fprintf('Y size: [%d, %d]\n', size(Y, 1), size(Y, 2));
fprintf('Y class: %s\n', class(Y));

% Convert Y to categorical if it's a cell array
if iscell(Y)
    fprintf('Converting Y from cell to categorical\n');
    Y = categorical(Y);
    fprintf('Y now has %d unique classes\n', numel(unique(Y)));
end

% STEP 2: Set up dimensions correctly
inputSize = 1024; % Number of features
numClasses = numel(unique(Y));
fprintf('Input size: %d, Number of classes: %d\n', inputSize, numClasses);

% STEP 3: First try a direct test to diagnose the issue
% fprintf('\n==== TESTING DIRECT NETWORK TRAINING ====\n');
% try
%     % The key fix: X_test should NOT be transposed here
%     % MATLAB's trainNetwork expects X as [samples x features] when Y is [samples x 1]
%     X_test = X;  % Keep as [samples x features]
% 
%     simple_layers = [
%         featureInputLayer(inputSize)
%         fullyConnectedLayer(numClasses)
%         softmaxLayer
%         classificationLayer];
% 
%     simple_options = trainingOptions('adam', ...
%         'MaxEpochs', 5, ...
%         'MiniBatchSize', 64, ...
%         'Verbose', false);
% 
%     fprintf('X_test dimensions: [%d, %d]\n', size(X_test, 1), size(X_test, 2));
%     fprintf('Y dimensions: [%d, %d]\n', size(Y, 1), size(Y, 2));
% 
%     % Try training with a small subset first
%     subset_size = min(1000, size(X, 1));
%     X_subset = X_test(1:subset_size, :);  % Fixed indexing for subset
%     Y_subset = Y(1:subset_size);
% 
%     fprintf('Training test network with subset (%d samples)...\n', subset_size);
%     test_net = trainNetwork(X_subset, Y_subset, simple_layers, simple_options);
%     fprintf('Test training successful!\n');
% catch ME
%     fprintf('Test training failed: %s\n', ME.message);
%     fprintf('This indicates an issue with the basic data format.\n');
%     return;  % Stop execution if basic test fails
% end

% STEP 4: Define optimization variables (simplified)
fprintf('\n==== SETTING UP BAYESIAN OPTIMIZATION ====\n');
optimVars = [
    optimizableVariable('numHiddenLayers',[1 5],'Type','integer')
    optimizableVariable('numNeurons',[64 384],'Type','integer')
    optimizableVariable('initialLearnRate',[1e-4 1e-2],'Transform','log')
    optimizableVariable('L2Regularization',[1e-5 1e-2],'Transform','log')
    optimizableVariable('miniBatchSize',[16 128],'Type','integer')
    optimizableVariable('dropoutRate',[0 0.5])
];

% STEP 5: Objective function for Bayesian optimization
ObjFcn = @(params) deepnetObjectiveFcn(X, Y, inputSize, numClasses, params);

% STEP 6: Run Bayesian optimization with minimal evaluations
fprintf('Starting Bayesian optimization...\n');
results = bayesopt(ObjFcn, optimVars, ...
    'MaxObjectiveEvaluations', 300, ...  % num iterations
    'IsObjectiveDeterministic', false, ...
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'UseParallel', false);

%%%%%%%% logging optimization stuff

% Plot the minimum objective value over iterations
figure('Position', [100, 100, 800, 500]);

% Plot 1: Minimum objective vs iterations
subplot(2,1,1);
plot(results.ObjectiveMinimumTrace, 'b-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Minimum Classification Error');
title('Best Classification Error vs. Iteration');
grid on;

% Plot 2: Objective value for each evaluation
subplot(2,1,2);
iterationNumbers = 1:results.NumObjectiveEvaluations;
scatter(iterationNumbers, results.ObjectiveTrace, 70, 'filled');
hold on;
plot(iterationNumbers, results.ObjectiveTrace, 'k--', 'LineWidth', 1);
xlabel('Iteration');
ylabel('Classification Error');
title('Classification Error for Each Evaluation');
grid on;

% Save the figure
saveas(gcf, 'training_results/optimization_results.png');
fprintf('Optimization plots saved to optimization_results.png\n');

% You can also display detailed information about the best result
fprintf('\n===== OPTIMIZATION RESULTS =====\n');
fprintf('Best classification error: %.4f\n', results.MinObjective);
fprintf('Best hyperparameters:\n');
disp(results.XAtMinObjective);

% You can also create a table with all evaluations sorted by performance

% evalTable = [results.ObjectiveTrace', array2table(results.XTrace)];
% 
% evalTable = array2table(results.XTrace);
% evalTable.ClassificationError = results.ObjectiveTrace';
% evalTable = movevars(evalTable, 'ClassificationError', 'Before', 1);
% % --------------
% evalTable.Properties.VariableNames{1} = 'ClassificationError';
% sortedEvals = sortrows(evalTable, 'ClassificationError');
% fprintf('\nTop 3 configurations:\n');
% disp(sortedEvals(1:min(3, height(sortedEvals)), :));

% Create model importance plot
% figure('Position', [100, 100, 600, 400]);
% imp = results.VariableImportance;
% [sortedImp, idx] = sort(imp, 'descend');
% paramNames = results.VariableDescriptions(idx);
% 
% barh(sortedImp);
% set(gca, 'YTick', 1:length(paramNames), 'YTickLabel', paramNames);
% xlabel('Relative Importance');
% title('Hyperparameter Importance');
% grid on;
% saveas(gcf, 'hyperparameter_importance.png');
% fprintf('Hyperparameter importance plot saved to hyperparameter_importance.png\n');

% ------------------------
% Save best model
bestParams = results.XAtMinObjective;
[~, bestNet] = deepnetObjectiveFcn(X, Y, inputSize, numClasses, bestParams);
save('training_results/trained_deep_model.mat', 'bestNet', 'bestParams');

fprintf('IMP NOTE: The order of labels for the saved model is:\n');
disp(bestNet.Layers(end).Classes);

fprintf('Training complete. Best model saved.\n');
