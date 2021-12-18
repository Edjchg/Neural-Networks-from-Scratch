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

# Calculating outputs using loops:
# Modeling 1 neuron:
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1]
bias = 2
neuron = Neuron(inputs, weights, bias, False)
neuronOutput = neuron.getOutput()
print(neuronOutput)

# Modeling 3 neurons:
inputs = [1, 2, 3, 2.5]
weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
weights = [weights1,
           weights2,
           weights3]

bias1 = 2
bias2 = 3
bias3 = 0.5
biases = [bias1,
          bias2,
          bias3]

layerOutputs = []

for weight, bias in zip(weights, biases):
    if 0 < len(weight) or 0 < len(bias):
        newNeuron = Neuron(inputs, weight, bias, False)
        layerOutputs.append(newNeuron.getOutput())

print(layerOutputs)

# Calculating outputs with vector math:
# Modeling 1 neuron:
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1]
bias = 2
neuron = Neuron(inputs, weights, bias, True)
neuronOutput = neuron.getOutput()
print(neuronOutput)
