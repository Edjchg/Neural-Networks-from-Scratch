import math
import numpy as np

class ActivationSoftmax:
    # Exponentietion + Normalization = Softmax activation function.
    def __init__(self):
        self.outputs = []

    def getOutputs(self):
        return self.outputs

    def forward(self, inputs):
        # Exponentiation: managing the signs
        # Exponentiation may derive in overflow, to prevent this, with substract the larger value to the arrays.
        exponentialValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalization:
        # Axis = 1: to sum the rows from "layer_outputs", the result will be: [a, b,c, ..., n]
        # Keepdims = true: to keep the result in a column
        probabilities = exponentialValues / np.sum(exponentialValues, axis=1, keepdims=True)
        self.outputs = probabilities
