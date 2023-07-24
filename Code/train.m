trainData = readtable('Data_files\train_data.xlsx'); % training data

tic;

[trainedClassifier, validationAccuracy] = trainTreeClassifier(trainData); % training the model on above data (specify respective function name of model)

training_time = toc;

testData = readtable('Data_files\test_data_1.xlsx', 'ReadVariableNames', false); % reading test data into a table

features = testData(2:end, 1:end-1); 
labels = testData{2:end, end}; % actual labels stored in this cell for comparison with predictions

tic;

[yfit, scores] = trainedClassifier.predictFcn(features); % testing the data

testing_time = toc;

yfit; % results of predictions stored in this list

% yfit
% 
% scores

sumVector = sum(scores, 1);

% sumVector

maxpos = find(sumVector == max(sumVector));

% maxpos

label_order = ["leaning_forward", "leaning_left", "leaning_leftlegcrossed", "leaning_right", "leaning_rightlegcrossed", "slouch", "straight"];

prediction = label_order(maxpos);

prediction