import numpy as np


class ActivationReLU:
    def __init__(self):
        np.random.seed(0)
        self.output = []

    def forward(self, inputs):
        """
        The following code is the activation function, which can be
        replaced by the np.maximum function:

        X, y = inputs.shape
        self.output = np.zeros((X,y))
        for i in range(X):
            for j in range(y):
                if inputs[i, j] > 0:
                    self.output[i,j] = inputs[i, j]
                else:
                    self.output[i, j] = 0
        """
        self.output = np.maximum(0, inputs)

    def getOutputs(self):
        return self.output
