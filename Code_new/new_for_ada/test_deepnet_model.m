% test_deepnet_model.m

% === Configuration ===
modelPath = 'training_results/trained_deep_model.mat';   % Path to .mat model file
testCSV   = 'test_7_3label.csv';     % Path to test CSV file
outputDir = 'testing_results';       % Folder to save outputs
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

fprintf('Loading trained model from: %s\n', modelPath);
load(modelPath, 'bestNet');  % loads 'bestNet'

% === Load test data ===
fprintf('Reading test data from: %s\n', testCSV);
testData = readtable(testCSV);

% === Extract features and labels ===
XTest = testData{:, 1:1024};
YTest = testData.three_label;

% Ensure labels are categorical
if iscell(YTest)
    YTest = categorical(YTest);
elseif isnumeric(YTest)
    YTest = categorical(string(YTest));
end

% === Predict using the trained model ===
YPred = classify(bestNet, XTest);
scores = predict(bestNet, XTest);  % For ROC

% Ensure categories and score columns match
classNames = categories(YTest);  % e.g., ["correct", "incorrect", "not_sitting"]
[~, labelIndices] = ismember(classNames, bestNet.Layers(end).Classes);
scores = scores(:, labelIndices);

% === Accuracy ===
acc = mean(YPred == YTest);
fprintf('Test Accuracy: %.2f%%\n', acc * 100);

% === Confusion Matrix ===
figure;
cm = confusionchart(YTest, YPred);
cm.Title = 'Confusion Matrix';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

savefig(fullfile(outputDir, 'confusion_matrix.fig'));
saveas(gcf, fullfile(outputDir, 'confusion_matrix.png'));
fprintf('Confusion matrix saved.\n');

% === ROC Curve with Operating Points ===
classNames = categories(YTest);
numClasses = numel(classNames);

% One-hot encoding of true labels
YTestOneHot = zeros(numel(YTest), numClasses);
for i = 1:numClasses
    YTestOneHot(:,i) = YTest == classNames{i};
end

% ROC Curve for each class
figure; hold on;
colors = lines(numClasses);
legends = {};

for i = 1:numClasses
    [X, Y, ~, AUC] = perfcurve(YTest, scores(:,i), classNames{i});
    
    % Plot ROC
    plot(X, Y, 'Color', colors(i,:), 'LineWidth', 2);
    legends{end+1} = sprintf('%s (AUC = %.3f)', classNames{i}, AUC);
    
    % Add operating point
    preds = vec2ind(scores')'; % predicted class indices
    operating_idx = YTest == classNames{i}; % true class i
    threshold_preds = preds == i;           % predicted class i
    TP = sum(operating_idx & threshold_preds);
    FP = sum(~operating_idx & threshold_preds);
    FN = sum(operating_idx & ~threshold_preds);
    TN = sum(~operating_idx & ~threshold_preds);

    FPR = FP / (FP + TN);
    TPR = TP / (TP + FN);

    plot(FPR, TPR, 'o', 'Color', colors(i,:), 'MarkerSize', 8, 'MarkerFaceColor', colors(i,:));
end

xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curves with Operating Points');
legend(legends, 'Location', 'SouthEast');
grid on;

savefig(fullfile(outputDir, 'roc_with_op.fig'));
saveas(gcf, fullfile(outputDir, 'roc_with_op.png'));
fprintf('ROC curve with operating points saved.\n');


