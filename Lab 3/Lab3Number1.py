import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

# Load the Iris dataset from sklearn
iris = load_iris()

# Extract features (input array X) and target (target array T)
X = iris.data
T = iris.target.reshape(-1, 1)  # Reshape to make it a column vector

# Perform one-hot encoding for the target variables
encoder = OneHotEncoder()
targets = encoder.fit_transform(T).toarray()

# Adjust Layers as needed
layers = [X.shape[1], 20, targets.shape[1]] 
 
# Test Inputs and Layers =================================================================
# X = [[2,1]]
# layers = [2, 3, 2]  
# targets = [0 , 1]

# X = [[2,1]]
# layers = [2, 2, 3, 2]  
# targets = [0 , 1]
# Test Inputs and Layers =================================================================



def nut_network(X, targets ,layers):

    learning_rate = 0.1
    epochs = 300
    epsilon = 3
    lossSum = float('inf')

    # Initialize weights
    weights = []
    for i in range(len(layers) - 1):
        weight_matrix = np.random.randn(layers[i], layers[i + 1])
        weights.append(weight_matrix)

    # Initialize biases
    biases = []
    for i in range(len(layers) - 1):
        bias_matrix = np.random.randn(layers[i + 1])
        biases.append(bias_matrix)

    # Test weights  =================================================================
    # weights = [[[2,1,-1],[-2,-4,1]],[[-2,3],[3,-1],[5,0]]]
    # # Test biases
    # biases = [[1,-1,2],[0,-1]]
    # weights = [[[2,1],[-2,-4]],[[-2,3,1],[-2,3,1]],[[-2,3],[3,-1],[3,-1]]]
    # # Test biases
    # biases = [[1,-2],[0,-1,1],[0,-1]]
    # Test weights  =================================================================
    
    def forwardProp(input):
        
        def sigmoid(x): 
            return 1 / (1 + np.exp(-x))

        def calculateA(A, layer):
            return sigmoid(np.dot(A,weights[layer]) + biases[layer])
        
        activations = []
        
        # calculate activations of each layer & input layer
        A = calculateA(input, 0)
        activations.append(A)

        for i in range(len(layers) - 1):
            if (i != 0):
                A = calculateA(A, i)
                activations.append(A)

        # return a 2d arrya of all activation values
        return activations
 
    def calcDeltas(activations, targets, weights):
        deltas=[]
        # going backwards from the output layer
        for layer in reversed(range(0 ,len(activations))):
            # if its the output layer, use the simple function
            if (layer == len(activations)-1):
                delta = (activations[layer]-targets)*activations[layer]*(1-activations[layer])
                # print()
                # print(layer)
                # print(delta)
                deltas.insert(0,delta)

            # if its not the last layer use upper layers delta to calculate delta
            else:
                sumOfDeltaXWeights = 0
                for weight in weights[layer+1]:
                    sumOfDeltaXWeights += np.dot(deltas[0],weight)

                delta = sumOfDeltaXWeights*activations[layer]*(1-activations[layer])
                # print()
                # print(layer)
                # print(delta)
                deltas.insert(0,delta)
        
        # return a 2d array of all deltas
        return deltas
    
    def updateWeights(weights, deltas, activations, input):
        # for all layers, update the weights
        for layer in range(0 ,len(activations)):
            # print()
            # print(layer)
            # print(weights[layer])
            # print(deltas[0])

            # if its the first layer, use the input values in the calculation
            if (layer == 0):
                 for node in range(len(weights[0])):
                    # print(weights[0][node], " - " , " N * ", deltas[0], " * ", input[node])
                    weights[0][node] =  weights[0][node] - learning_rate*(deltas[0]*input[node])
            
            # if its not the first layer, use the activaion values in the calculation
            else:
                for node in range(len(weights[layer])):
                    # print(weights[layer][node], " - " , " N * ", deltas[layer], " * ", activations[layer-1][node])
                    weights[layer][node] =  weights[layer][node] - learning_rate*(deltas[layer]*activations[layer-1][node])
            # print(weights[layer])
        
        return weights
    
    def updateBiases(biases, deltas, activations):
        # for each layer in the network
        for layer in range(0 ,len(activations)):
            # print()
            # print(layer)
            # print(biases[layer])
            # print(deltas[layer])

            # update the biases
            for node in range(len(biases[layer])):
                # print(biases[layer][node], " - " , " N * ", deltas[layer][node])
                biases[layer][node] =  biases[layer][node] - learning_rate*deltas[layer][node]
            # print(biases[layer])
        
        return biases
    
    def lossFunction(actualOutputs, targetOutputs):
        # calculate the loss function over all the data points
        lossSum = 0
        for output in range(len(actualOutputs)):
            lossSum += np.sum((actualOutputs[output] - targetOutputs[output])**2)

        return lossSum/2
    
    def predictOutput(X):
        # get the predicted outputs from the input
        outputs = []
        for input in X:
            output = forwardProp(input)
            outputs.append(output[-1])

        return outputs
    
    def formatOutputs(outputs):
        # reformat the outputs to suit the targets
        oneHotOutputs = []
        for output in outputs:
            max = np.argmax(output)
            oneHotOutput = np.zeros_like(output)
            oneHotOutput[max] = 1
            
            oneHotOutputs.append(oneHotOutput)
        
        return oneHotOutputs

    def calculateAccuracy(actualOutputs, targetOutputs):
        # calculate the accuracy of the predicted data
        matches = 0
        samples = len(actualOutputs)
        
        for i in range(samples):
            if np.array_equal(actualOutputs[i], targetOutputs[i]):
                matches += 1
        
        accuracy = (matches / samples) * 100
        return accuracy
    
    data = list(zip(X, targets))
    epoch = 0
    while ((lossSum > epsilon) and (epoch < epochs)):
        np.random.shuffle(data)
        for input, target in data:

            # print()
            # print("======================Activations==========================")
            # print()

            activations = forwardProp(input)

            # print()
            # print("======================Deltas==========================")
            # print()

            deltas = calcDeltas(activations, target, weights)

            # print()
            # print("======================Weights==========================")
            # print()


            weights = updateWeights(weights, deltas, activations, input)

            # print()
            # print("======================Biases==========================")
            # print()

            biases = updateBiases(biases, deltas, activations)

        outputs = predictOutput(X)
        lossSum = lossFunction(outputs,targets)
        epoch+=1
        print("Epcoh:" , epoch, "with loss", lossSum)
    
    outputs = predictOutput(X)
    outputs = formatOutputs(outputs)
    accuarcy = calculateAccuracy(outputs, targets)

    print("=======================================================================================")
    print("Model predecting with: " , accuarcy, " percent accuracy")
    print("=======================================================================================")

    # print("Weights: ", weights)
    # print("Biases: ", biases)

 
nut_network(X, targets, layers)