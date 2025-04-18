% === Configuration ===
modelPath = 'training_results/trained_deep_model.mat';
testCSV   = 'test_7_9label.csv';
outputDir = 'testing_results';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

fprintf('Loading trained model from: %s\n', modelPath);
load(modelPath, 'bestNet');

% === Load test data ===
fprintf('Reading test data from: %s\n', testCSV);
testData = readtable(testCSV);

% === Extract features and labels ===
XTest = testData{:, 1:1024};
YTest = testData.posture;

% Ensure labels are categorical
if iscell(YTest)
    YTest = categorical(YTest);
elseif isnumeric(YTest)
    YTest = categorical(string(YTest));
end

% === Predict using the trained model ===
YPred = classify(bestNet, XTest);
scores = predict(bestNet, XTest);

% Ensure categories and score columns match
classNames = categories(YTest);
[~, labelIndices] = ismember(classNames, bestNet.Layers(end).Classes);
scores = scores(:, labelIndices);

% === Accuracy ===
acc = mean(YPred == YTest);
fprintf('Test Accuracy: %.2f%%\n', acc * 100);

% === Save data for Confusion Matrix and ROC Curve plotting elsewhere ===
% Convert categorical to cellstr for table compatibility
trueLabels = cellstr(YTest);
predLabels = cellstr(YPred);

% Create the base table
resultTable = table(trueLabels, predLabels, 'VariableNames', {'TrueLabel', 'PredictedLabel'});

% Add score columns one by one
for i = 1:numel(classNames)
    resultTable.(classNames{i}) = scores(:, i);
end

csvwritePath = fullfile(outputDir, 'predictions_scores.csv');
writetable(resultTable, csvwritePath);
fprintf('Predictions and scores saved to: %s\n', csvwritePath);

% === Confusion Matrix Plot (for local inspection) ===
figure;
cm = confusionchart(YTest, YPred);
cm.Title = 'Confusion Matrix';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

savefig(fullfile(outputDir, 'confusion_matrix.fig'));
saveas(gcf, fullfile(outputDir, 'confusion_matrix.png'));
fprintf('Confusion matrix saved.\n');

% === ROC Curve with Operating Points (for local inspection) ===
numClasses = numel(classNames);

% One-hot encoding of true labels
YTestOneHot = zeros(numel(YTest), numClasses);
for i = 1:numClasses
    YTestOneHot(:,i) = YTest == classNames{i};
end

figure; hold on;
colors = lines(numClasses);
legends = {};

for i = 1:numClasses
    [X, Y, ~, AUC] = perfcurve(YTest, scores(:,i), classNames{i});

    plot(X, Y, 'Color', colors(i,:), 'LineWidth', 2);
    legends{end+1} = sprintf('%s (AUC = %.3f)', classNames{i}, AUC);

    preds = vec2ind(scores')';
    operating_idx = YTest == classNames{i};
    threshold_preds = preds == i;
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
