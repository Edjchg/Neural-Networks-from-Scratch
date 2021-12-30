import sys
import matplotlib
import matplotlib.pyplot as plt

from Neuron import *
from LayerDense import *
from ActivationReLU import *
from ActivationSoftmax import *


def show_information():
    info = "------------------------------------------------\n"
    info += "General Information:\n"
    info += "Description:\tNeural Networks from Scratch.\n"
    info += "Author:\tEdgar Chaves\n"
    info += "Year:\t2021-2022\n"
    info += "------------------------------------------------\n"
    print(info)


def show_references():
    references = "------------------------------------------------\n"
    references += "References: \n"
    references += "Sentdex: \thttps://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3" \
                  "&index=1 \n "
    references += "------------------------------------------------\n"
    print(references)


def show_libraries():
    libraries = "------------------------------------------------\n"
    libraries += "Used libraries:\n"
    libraries += "Python:" + "\t" + str(sys.version) + "\n"
    libraries += "Numpy:" + "\t" + str(np.__version__) + "\n"
    libraries += "Matplotlib:" + "\t" + matplotlib.__version__ + "\n"
    libraries += "------------------------------------------------\n"
    print(libraries)


def neuronExperiments():
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


def layerExperiments():
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


def createDataSet(points, classes):
    np.random.seed(0)
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')

    for classNumber in range(classes):
        ix = range(points * classNumber, points * (classNumber + 1))
        r = np.linspace(0, 1, points)  # radius
        t = np.linspace(classNumber * 4, (classNumber + 1) * 4, points) + np.random.randn(points) * 2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = classNumber
        return X, y


def plotDataSet(X, y):
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
    plt.show()


def ReLUExperiments(X):
    NUMBER_OF_FEATURES = 2
    NUMBER_OF_NEURONS = 4
    layer1 = LayerDense(NUMBER_OF_FEATURES, NUMBER_OF_NEURONS)
    activation1 = ActivationReLU()

    layer1.forward(X)
    print(layer1.getOutput())
    activation1.forward(layer1.getOutput())
    print(activation1.getOutputs())


def SoftmaxExperiments(X):
    NUMBER_OF_FEATURES = 2
    NUMBER_OF_NEURONS = 3
    dense1 = LayerDense(NUMBER_OF_FEATURES, NUMBER_OF_NEURONS)
    activation1 = ActivationReLU()

    NUMBER_OF_FEATURES = 3
    NUMBER_OF_NEURONS = 3
    dense2 = LayerDense(NUMBER_OF_FEATURES, NUMBER_OF_NEURONS)
    activation2 = ActivationSoftmax()

    dense1.forward(X)
    activation1.forward(dense1.getOutput())

    dense2.forward(activation1.getOutputs())
    activation2.forward(dense2.getOutput())

    print(activation2.getOutputs())

if __name__ == '__main__':
    show_information()
    show_libraries()
    show_references()
    print("*************************************************")
    neuronExperiments()
    print("*************************************************")
    layerExperiments()
    print("*************************************************")
    print("Creating dataset...")
    X, y = createDataSet(100, 3)
    print("Plotting dataset...")
    plotDataSet(X, y)
    print("*************************************************")
    ReLUExperiments(X)
    print("*************************************************")
    SoftmaxExperiments(X)
