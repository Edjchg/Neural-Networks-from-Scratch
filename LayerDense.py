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


# Input data:
X = [[1, 2, 3, 2.5],
     [2, 5, -1, 2],
     [-1.5, 2.7, 3.3, -0.8]]

# The first layer:
layer1 = LayerDense(4, 5)
layer1.forward(X)

# The output from layer 1 is going to be the input of the layer 2:
layer2 = LayerDense(5, 2)
layer2.forward(layer1.getOutput())

finalOutput = layer2.getOutput()
print(finalOutput)