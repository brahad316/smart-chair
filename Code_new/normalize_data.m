% Read the CSV file
data = readtable('test_7.csv');

% Extract numeric columns (first 1027 columns)
numeric_data = data{:, 1:1027};

% Normalize each column (Min-Max normalization)
numeric_data = (numeric_data - min(numeric_data)) ./ (max(numeric_data) - min(numeric_data));

% Replace NaN values (caused by division by zero) with zero
numeric_data(isnan(numeric_data)) = 0;

% Convert back to table while preserving column names
data(:, 1:1027) = array2table(numeric_data, 'VariableNames', data.Properties.VariableNames(1:1027));

% Write to a new CSV file
writetable(data, 'test_7_normalized.csv');
