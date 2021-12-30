import numpy as np


# Modeling a neuron:

class Neuron:
    def __init__(self, inputs, weights, bias, vectorizedCalc):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.output = 0

        if vectorizedCalc:
            self.calculateOutputDotProduct()
        else:
            self.calculateOutputsLoops()

    def calculateOutputsLoops(self):
        inputsSize = len(self.inputs)
        weightsSize = len(self.weights)
        if inputsSize == weightsSize:
            for i in range(0, inputsSize):
                self.output += self.inputs[i] * self.weights[i]
            self.output += self.bias

    def calculateOutputDotProduct(self):
        self.output = np.dot(self.weights, self.inputs) + self.bias

    def getOutput(self):
        return self.output



