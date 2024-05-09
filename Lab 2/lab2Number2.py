import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = load_iris()
X = iris.data  # Input features
T = iris.target  # Target labels

# Convert target labels to one-hot encoding
encoder = OneHotEncoder(sparse=False)
T_onehot = encoder.fit_transform(T.reshape(-1, 1))

# Split the dataset into training and test sets
X_train, X_test, T_train, T_test = train_test_split(X, T_onehot, test_size=0.2, random_state=42)

# Define the neural network architecture
input_size = X_train.shape[1]
hidden_size = 10  # Number of neurons in the hidden layer
output_size = T_train.shape[1]  # Number of output classes
learning_rate = 0.01
epochs = 1000

# Initialize weights and biases
weights_input_hidden = np.random.randn(input_size, hidden_size)
bias_hidden = np.random.randn(hidden_size)
weights_hidden_output = np.random.randn(hidden_size, output_size)
bias_output = np.random.randn(output_size)

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Training the neural network
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X_train, weights_input_hidden) + bias_hidden
    hidden_layer_activation = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_activation, weights_hidden_output) + bias_output
    output_layer_activation = sigmoid(output_layer_input)
    
    # Backpropagation
    error = T_train - output_layer_activation
    d_output = error * sigmoid_derivative(output_layer_activation)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)
    
    # Update weights and biases
    weights_hidden_output += hidden_layer_activation.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0) * learning_rate
    weights_input_hidden += X_train.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0) * learning_rate

# Testing the trained model
hidden_layer_input = np.dot(X_test, weights_input_hidden) + bias_hidden
hidden_layer_activation = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_activation, weights_hidden_output) + bias_output
output_layer_activation = sigmoid(output_layer_input)

# Convert output probabilities to predicted labels
predicted_labels = np.argmax(output_layer_activation, axis=1)
true_labels = np.argmax(T_test, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_labels == true_labels)
print("Accuracy:", accuracy)
