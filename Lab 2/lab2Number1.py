import math
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder


# Function to for activations - Sigmoid
def sigmoid(x):
    return 1/(1+(math.e**-x))

# Function to perform the feedforward into the network
def feedforward(inputValues, WeightsInputHidden, BiasHidden, weightsOutputHidden, biasOutput):

    hiddenLayerInput = np.dot(inputValues, WeightsInputHidden) + BiasHidden
    hiddenLayerActivation = sigmoid(hiddenLayerInput)

    outputLayerInput = np.dot(hiddenLayerActivation, weightsOutputHidden) + biasOutput
    outputLayerActivation = sigmoid(outputLayerInput)

    return outputLayerActivation


inputValues = np.array([0.2, 0.3, 0.4])
WeightsInputHidden = np.random.randn(3, 4)
biasHidden = np.random.randn(4)
weightsOutputHidden = np.random.randn(4, 3)
biasOutput = np.random.randn(1)

output = feedforward(inputValues, inputValues, biasHidden, weightsOutputHidden, biasOutput)

print(output)

