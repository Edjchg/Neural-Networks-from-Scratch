import numpy as np


# Modeling Layers:

class LayerDense:
    def __init__(self, numberOfInputs, numberOfNeurons):
        self.outputs = 0
        np.random.seed(0)
        # Gaussian distribution around 0:
        self.weights = 0.10 * np.random.randn(numberOfInputs, numberOfNeurons)
        self.biases = np.zeros((1, numberOfNeurons))

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def getOutput(self):
        return self.outputs
    