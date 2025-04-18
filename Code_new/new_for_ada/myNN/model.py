#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    18-Apr-2025 16:45:16

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    input_unnormalized = keras.Input(shape=(1024,), name="input_unnormalized")
    input = ZScoreLayer((1024,), name="input_")(input_unnormalized)
    fc_1 = layers.Dense(189, name="fc_1_")(input)
    relu_1 = layers.ReLU()(fc_1)
    fc_2 = layers.Dense(189, name="fc_2_")(relu_1)
    relu_2 = layers.ReLU()(fc_2)
    fc_3 = layers.Dense(3, name="fc_3_")(relu_2)
    softmax = layers.Softmax()(fc_3)

    model = keras.Model(inputs=[input_unnormalized], outputs=[softmax])
    return model

## Helper layers:

class ZScoreLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name=None):
        super(ZScoreLayer, self).__init__(name=name)
        self.mean = tf.Variable(initial_value=tf.zeros(shape), trainable=False)
        self.std = tf.Variable(initial_value=tf.zeros(shape), trainable=False)

    def call(self, input):
        # Compute z-score of input
        return (input - self.mean)/self.std

