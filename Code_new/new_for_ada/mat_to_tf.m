% Load your trained model
load('training_results/trained_deep_model.mat', 'bestNet');

% Convert to a layerGraph
lgraph = layerGraph(bestNet.Layers);

% Remove the final classification output layer
lgraph = removeLayers(lgraph, 'output');

% Now create a dlnetwork from the cleaned-up layer graph
dlnet = dlnetwork(lgraph);

% myNN is in the desired SavedModel format
exportNetworkToTensorFlow(dlnet, "myNN");