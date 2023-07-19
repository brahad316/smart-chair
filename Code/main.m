trainData = readtable('train_data.xlsx'); % training data

tic;

[trainedClassifier, validationAccuracy] = trainTreeClassifier(trainData); % training the model on above data (specify respective function name of model)

training_time = toc;

testData = readtable('test_data.xlsx', 'ReadVariableNames', false); % reading test data into a table

features = testData(2:end, 1:end-1); 
labels = testData{2:end, end}; % actual labels stored in this cell for comparison with predictions

tic;

[yfit, scores] = trainedClassifier.predictFcn(features); % testing the data

testing_time = toc;

yfit; % results of predictions stored in this list

sideBySide = horzcat(yfit, labels) % show prediction vs actual label

disp(['training time: ', num2str(training_time), ' seconds']);
disp(['testing time: ', num2str(testing_time), ' seconds']);