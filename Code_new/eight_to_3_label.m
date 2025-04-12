% Read the CSV file into a table
data = readtable('test_7_normalized.csv');

% Define which postures are considered "correct"
correctPostures = {'straight', 'cross_legged'};

% Create a new column 'three_label' based on the 'posture' column
data.three_label = repmat("incorrect", height(data), 1);  % Default label

% Assign 'correct' to rows where posture is 'straight' or 'cross_legged'
data.three_label(ismember(data.posture, correctPostures)) = "correct";

% Write the modified table to a new CSV file
writetable(data, 'test_7_normalized_3label.csv');
