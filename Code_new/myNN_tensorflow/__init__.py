#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    12-Apr-2025 00:51:59

from . import model
from .model import ZScoreLayer
import os

def load_model(load_weights=True, debug=False):
    m = model.create_model()
    if load_weights:
        loadWeights(m, debug=debug)
    return m

## Utility functions:

import tensorflow as tf
import h5py

def loadWeights(model, filename=os.path.join(__package__, "weights.h5"), debug=False):
    with h5py.File(filename, 'r') as f:
        # Every layer is an h5 group. Ignore non-groups (such as /0)
        for g in f:
            if isinstance(f[g], h5py.Group):
                group = f[g]
                layerName = group.attrs['Name']
                numVars = int(group.attrs['NumVars'])
                if debug:
                    print("layerName:", layerName)
                    print("    numVars:", numVars)
                # Find the layer index from its name
                layerIdx = layerNum(model, layerName)
                layer = model.layers[layerIdx]
                if debug:
                    print("    layerIdx=", layerIdx)
                # Every weight is an h5 dataset in the layer group
                weightList = [0]*numVars
                for d in group:
                    dataset = group[d]
                    varName = dataset.attrs['Name']
                    shp     = intList(dataset.attrs['Shape'])
                    weightNum = int(dataset.attrs['WeightNum'])
                    if debug:
                        print("    varName:", varName)
                        print("        shp:", shp)
                        print("        weightNum:", weightNum)
                    weightList[weightNum] = tf.constant(dataset[()], shape=shp)
                # Assign weights into the layer
                if isinstance(layer, ZScoreLayer):
                    if debug:
                        print("Assigning to ZScoreLayer attributes...")
                    if numVars != 2:
                        raise ValueError(f"Expected 2 variables for ZScoreLayer, got {numVars}")
                    layer.mean.assign(weightList[0])
                    layer.std.assign(weightList[1])
                    if debug:
                        print("Assignment successful.")
                        print("Set mean:", layer.mean)
                        print("Set std:", layer.std)
                else:
                    for w in range(numVars):
                        if debug:
                            print("Copying variable of shape:")
                            print(weightList[w].shape)
                        layer.variables[w].assign(weightList[w])
                        if debug:
                            print("Assignment successful.")
                            print("Set variable value:")
                            print(layer.variables[w])
                # Finalize layer state
                if hasattr(layer, 'finalize_state'):
                    layer.finalize_state()

def layerNum(model, layerName):
    layers = model.layers
    for i in range(len(layers)):
        if layerName == layers[i].name:
            return i
    print("\nWEIGHT LOADING FAILED. MODEL DOES NOT CONTAIN LAYER WITH NAME:", layerName)
    return -1

def intList(myList): 
    return list(map(int, myList))
