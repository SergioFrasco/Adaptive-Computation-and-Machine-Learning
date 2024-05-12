import sys
import math

# Assign the number of clusters
clusterCenters = 3

# Provided dataset
data = [
    [0.22, 0.33], [0.45, 0.76], [0.73, 0.39], [0.25, 0.35],[0.51, 0.69], [0.69, 0.42], [0.41, 0.49], [0.15, 0.29],
    [0.81, 0.32], [0.50, 0.88], [0.23, 0.31], [0.77, 0.30], [0.56, 0.75], [0.11, 0.38], [0.81, 0.33], [0.59, 0.77],
    [0.10, 0.89], [0.55, 0.09], [0.75, 0.35], [0.44, 0.55]
]

# Calculate euclidean distance
def euclideanDistance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

# Assigning clusters
def clusterAssignment(dataPoints, clusterCenters):
    clusterSize = len(clusterCenters)
    clusters = []

    # append empty lists to clusters
    for i in range(clusterSize):
        cluster = []
        clusters.append(cluster)
    
    # check the distance between a cluster point and all data points
    for point in dataPoints:
        distToCenter = []
        
        for center in clusterCenters:
            distance = euclideanDistance(point, center)
            distToCenter.append(distance)
        
        # append the data point to its closest cluster point
        closestCluster = distToCenter.index(min(distToCenter))
        clusters[closestCluster].append(point)
    
    return clusters
# Update the cluster centers
def centerUpdate(clusters):
    centers = []
    for cluster in clusters:
        if len(cluster) > 0:
            xCenter = sum(point[0] for point in cluster) / len(cluster)
            yCenter = sum(point[1] for point in cluster) / len(cluster)
            centers.append([xCenter, yCenter])
        else:
            centers.append(None)
    return centers

# K means algorithm
def kmeans(data, k, initial_centers):
    centers = initial_centers
    while True:
        clusters = clusterAssignment(data, centers)
        new_centers = centerUpdate(clusters)
        if new_centers == centers:
            break
        centers = new_centers

    #  Remove empty clusters and centers
    non_empty_clusters = [cluster for cluster in clusters if len(cluster) > 0]
    non_empty_centers = [centroid for centroid, cluster in zip(centers, clusters) if len(cluster) > 0]

    return non_empty_clusters, non_empty_centers

# Sum of squares error
def ssError(clusters, centers):
    total_error = 0
    
    for cluster_index, cluster in enumerate(clusters):
        cluster_center = centers[cluster_index]
        
        for data_point in cluster:
            distance = euclideanDistance(data_point, cluster_center)
            squared_error = distance ** 2
            total_error += squared_error
    
    return total_error

# Read initial cluster centers from standard input
# Take in x and y for each cluster center
initial_centers = []
for i in range(clusterCenters):
    x = float(input())
    y = float(input())
    initial_centers.append([x, y])

clusters, centers = kmeans(data, clusterCenters, initial_centers) # Run k-means
error = ssError(clusters, centers) # Compute sum of squares error
print(round(error,4)) # Output result to 4 decimals