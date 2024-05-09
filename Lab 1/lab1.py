import numpy as np
import matplotlib.pyplot as plt

# data creation
def linear_function(x1, x2):
    return 2*x1 + 3*x2 - 1

dataset = np.array([])
targets = np.array([])

# Generate data
num_points = 50
for _ in range(num_points):
    r1 = np.random.uniform(-1,1)
    r2 = np.random.uniform(-1,1)
    dataset = np.append(dataset, [r1, r2])
    if linear_function(r1, r2) <= 0:
        target = 0
    else:
        target = 1
    targets = np.append(targets, target)

dataset = dataset.reshape(-1, 2)

indices_to_change = np.random.choice(len(targets), size=3, replace=False)
for idx in indices_to_change:
    targets[idx] = 1 - targets[idx]  # Change 0 to 1 or 1 to 0

# Print the modified targets
print("Modified Targets:")
print(targets)

# data plotting
print("dataset")
print(dataset)
print("targets")
print(targets)

dataset_0 = dataset[targets == 0]
dataset_1 = dataset[targets == 1]


def precum(dataset, targets):
    weights = np.random.uniform(0.0, 5.0, size=2)
    threshold = np.random.rand()
    learning_rate = 0.01

    def forwardProp(data_point):
        output = np.dot(data_point, weights)
        return int(output > threshold)

    epoch = 0
    totalLoss = 3
    while (epoch <= 10000 and totalLoss > 1):
        loss = 0
        for data_point in range(len(dataset)):

            # feed x into the perceptron, to get output y
            y = forwardProp(dataset[data_point])

            # update each weight wi using the rule: wi ← wi + η(t − y)xi
            weights = weights + learning_rate * (targets[data_point] - y)*dataset[data_point]

            # update the threshold θ using the rule: θ ← θ − η(t − y)
            threshold = threshold - learning_rate * (targets[data_point] - y)

            loss += np.abs(targets[data_point] - y)
        
        totalLoss = loss
        epoch += 1    

    return weights, threshold

weights, threshold = precum(dataset, targets)
plt.scatter(dataset_0[:, 0], dataset_0[:, 1], color='red')
plt.scatter(dataset_1[:, 0], dataset_1[:, 1], color='blue')
x_values = np.linspace(-1, 1, 100)
y_values = (-weights[0] * x_values + threshold) / weights[1]
plt.plot(x_values, y_values)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Scatter Plot of Dataset')
plt.legend()
plt.savefig('plot.png')
            