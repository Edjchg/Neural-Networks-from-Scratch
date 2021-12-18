import numpy as np


# Modeling Layers:

class LayerDense:
    def __init__(self, nInputs, nNeurons):
        np.random.seed(0)
        # Gaussian distribution around 0:
        self.weights = 0.10 * np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))


    def forward(self):
        pass


# Input data:
X = [[1, 2, 3, 2.5],
     [2, 5, -1, 2],
     [-1.5, 2.7, 3.3, -0.8]]
