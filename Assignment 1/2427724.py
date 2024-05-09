import numpy as np
 
def NerualNetwork(listOFInputs, outputTargets ,layerSizes):

    learning_rate = 0.1 # As specified
    weights = [] # Setting the weights
    biases = [] # setting biases

    # Utility function
    def sigmoid(x): 
        return 1 / (1 + np.exp(-x))
    
    # Utility function
    def calculateActivation(input, layer):
        return sigmoid(np.dot(input,weights[layer]) + biases[layer])

    # Weights
    for i in range(len(layerSizes) - 1):
        networkWeights = np.ones((layerSizes[i], layerSizes[i + 1]))
        weights.append(networkWeights)

    # Bias
    for i in range(len(layerSizes) - 1):
        networkBiases = np.ones(layerSizes[i + 1])
        biases.append(networkBiases)
    
    def loss(actual, targets):
        sumOfLosses = 0
        for point in range(len(actual)):
            sumOfLosses += np.sum((actual[point] - targets[point])**2)
        sumOfLosses = sumOfLosses/2

        return sumOfLosses
    
    def forwardPropagation(input):
        activations = []
        activation = calculateActivation(input, 0)
        activations.append(activation)

        for j in range(len(layerSizes) - 1):
            if (j != 0):
                activation = calculateActivation(activation, j)
                activations.append(activation)

        return activations

    def InitializeDeltas(activations, targets, weights):
        networkDeltas = []
        for layer in reversed(range(len(activations))):
            if layer == len(activations) - 1:
                delta = (activations[layer] - targets) * activations[layer] * (1 - activations[layer])
            else:
                delta_weight_sum = np.dot(networkDeltas[0], weights[layer + 1].T)
                delta = delta_weight_sum * activations[layer] * (1 - activations[layer])
            
            networkDeltas.insert(0, delta)

        return networkDeltas
    
    def calculateBias(biases, deltas, activations):
        # for each layer in the network
        for layer in range(0 ,len(activations)):
            # print()
            # print(layer)
            # print(biases[layer])
            # print(deltas[layer])

            # update the biases
            for node in range(len(biases[layer])):
                # print(biases[layer][node], " - " , learning_rate," * ", deltas[layer][node])
                biases[layer][node] =  biases[layer][node] - learning_rate*deltas[layer][node]
            # print(biases[layer])
        
        return biases
    
    def calculateWeight(netWeights, netDeltas, netActivations, netInput):

        for layer in range(0 ,len(netActivations)):
            # print()
            # print(layer)
            # print(weights[layer])
            # print(deltas[0])

            if layer == 0:
                 for node in range(len(netWeights[0])):
                    # print(weights[0][node], " - " ,learning_rate, " * ", deltas[0], " * ", input[node])
                    netWeights[0][node] =  netWeights[0][node] - learning_rate*(netDeltas[0]*netInput[node])
            
            else:
                for node in range(len(netWeights[layer])):
                    # print(weights[layer][node], " - " , learning_rate," * ", deltas[layer], " * ", activations[layer-1][node])
                    netWeights[layer][node] =  netWeights[layer][node] - learning_rate*(netDeltas[layer]*netActivations[layer-1][node])
            # print(weights[layer])
        return netWeights
    
    
    epoch = 0
    while (epoch < 1):
        for input in listOFInputs:
            activations = forwardPropagation(input)
            if epoch == 0 :
                sumOfLosses = loss(activations[1],outputTargets)
                print(round(sumOfLosses,4))

            deltas = InitializeDeltas(activations, outputTargets, weights)

            # print()
            # print("======================Deltas==========================")
            # print()
            # print('DELTAS',deltas)


            biases = calculateBias(biases, deltas, activations)
            weights = calculateWeight(weights, deltas, activations, input)

            # print('WEIGHTS',weights)
            
            # print()
            # print("======================Weights==========================")
            # print()
            
        epoch = epoch + 1
        
        # print()
        # print("======================Activations==========================")
        # print()

        activations = forwardPropagation(input)
        sumOfLosses = loss(activations[1],outputTargets)
        print(round(sumOfLosses,4))
 
temporaryList =[]
listOfInputs=[]
outputLayerTargets =[]
# Take in inputs
inputSize = 4
targetSize = 3

for j in range(inputSize):
    x = float(input())
    temporaryList.append(x)

listOfInputs.append(temporaryList)

for j in range(targetSize):
    x = float(input())
    outputLayerTargets.append(x)

layerSizes = [4, 8, 3] # as specified in the assignment

# Call the Nerual Netwrok
NerualNetwork(listOfInputs, outputLayerTargets, layerSizes)