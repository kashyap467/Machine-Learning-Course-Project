import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys
import colorsys

# Read from .mat file and load data
alSamplesData = scipy.io.loadmat('./AllSamples.mat')
inpData = np.array(alSamplesData['AllSamples'])
# plt.scatter(inpData[:,0], inpData[:,1])
# plt.title("Scatter plot of given samples")
# plt.show()

# INITIALIZATION OF CLUSTER CENTERS

# Strategy 1
def init_centers_strategy1(no_of_clusters, samples) : 
    '''
    Picks random cluster centers from given dataset
    '''
    # init_center_idxs = random.sample( range(samples.shape[0]), no_of_clusters )
    init_center_idxs = range(no_of_clusters)
    init_centroids = [samples[i] for i in init_center_idxs]
    return init_centroids


# Strategy 2
def init_centers_strategy2(no_of_clusters, samples) : 
    '''
    Picks first center randomly, chooses other centers with maximal average distance from previous ones
    '''
    # Picking first center randomly
    # init_center_idx = random.randrange(samples.shape[0])
    init_center_idx = no_of_clusters
    init_centroids = [samples[init_center_idx]]
    
    # Removing the sample selected as center from further comparisons 
    train = np.delete(samples, init_center_idx, 0)

    # Choosing other centers with maximal average distance from previous ones
    for _ in range(1,no_of_clusters) : 
        distFromXtoC = [[(np.linalg.norm(x-c)) for c in init_centroids] for x in train]
        newCenter = np.argmax([ row.mean() for row in np.array(distFromXtoC)])
        init_centroids.append(train[newCenter])
        train = np.delete(train, newCenter, 0)
        
    return init_centroids


# ASSIGN DATA SAMPLES TO THEIR NEAREST CLUSTERS

# Method to assign data samples to their nearest cluster centers based on euclidean distance
def assign_nearest_clusters(centroids, samples) :
    '''
    Assigns data samples to their nearest cluster centers based on euclidean distance
    '''
    assignedClusters = []
    for x in samples : 
        # find euclidean norms of each sample from all current cluster centers
        distToAllCenters = [ (np.linalg.norm(x-c)) for c in centroids ]
        # Assign the sample to the nearest cluster
        assignedClusters.append(np.argmin(distToAllCenters))
    return np.array(assignedClusters).reshape(-1,1)


# UPDATE CLUSTER CENTROIDS AS MEAN OF ALL SAMPLES BELONGING TO RESPECTIVE CLUSTERS

# Method to update cluster centers based on calculated mean of samples in that cluster
def update_centroids(clusters, samples) : 
    '''
    Updates cluster centers based on calculated mean of samples in that cluster
    '''
    updatedCentroids = []
    clusteredSamples = np.concatenate((samples, clusters), axis=1)
    # iterate over all clusters 
    for c in set(clusteredSamples[:,-1]) : 
        # Get the samples in current cluster
        present_cluster = clusteredSamples[ (clusteredSamples[:,-1] == c) ][:,:-1]
        # Calculate the mean of samples
        cluster_mean = present_cluster.mean(axis=0)
        # Set it as the new center for current cluster
        updatedCentroids.append(cluster_mean)
    return np.array(updatedCentroids).reshape(-1,2)


# Method to repeat steps 2,3 until the cluster centers converge ( don't change)
def assignAndUpdate(centroids, samples) : 
    '''
    Iteratively assigns clusters and updates the cluster centers until they converge
    '''
    centroidIterations = np.array([centroids])
    while (True) : 
        # Step 2 - Assignment of samples to clusters
        assignedClusters = assign_nearest_clusters(centroids, samples)
        # Step 3 - Update centers as mean of samples in that cluster
        updatedCentroids = update_centroids(assignedClusters, samples)
        # Check whether centers converged
        diff = np.absolute(np.subtract(updatedCentroids, centroids))
        result = (diff <= 0)
        if (np.all(result)) :   # If centers converged, break and return final centroids
            break
        centroids = updatedCentroids    # Else, continue the process until centers converge

    return centroids


# Method to calculate Objective Function values for given K (no of clusters)
def calc_ObjectiveFnValue(clusters, centroids, samples) :
    '''
    Calculates Objective Function values for given K (no.of.clusters)
    '''
    clusteredSamples = np.concatenate((samples, clusters), axis=1)
    objVal = 0
    for c in set(clusteredSamples[:,-1]) : 
        # Get the samples in current cluster
        present_cluster = clusteredSamples[clusteredSamples[:,-1] == c][:,:-1]
        # Square error value for all samples in the current cluster from its center
        dists = [np.linalg.norm(x - centroids[int(c)]) for x in present_cluster]
        # Objective function value (Sum of Square errors)
        objVal += np.sum(np.square(dists))

    print("Obj Val : {}".format(objVal))
    return objVal


# Method to implement complete K-Means algorithm with centroids initialization using either of two strategies
def kMeans(no_of_clusters, samples, init=1) : 
    
    # Step 1 - Initialization of cluster centers
    if init == 1 : 
        init_centroids = init_centers_strategy1(no_of_clusters, samples)
    elif init == 2 : 
        init_centroids = init_centers_strategy2(no_of_clusters, samples)
    else :
        print("Unknown parameter init={}. It should be 1 or 2".format(init))
        sys.exit(1)
    
    # Steps 2 & 3 until centers converge
    final_centroids = assignAndUpdate(init_centroids, samples)

    # Compute Objective function value for given K (no of clusters)
    finalClusters_idx = assign_nearest_clusters(final_centroids, samples)
    return ( calc_ObjectiveFnValue(finalClusters_idx, final_centroids, samples) )


# IMPLEMENTING K-MEANS ALGORITHM WITH CENTROIDS INITIALIZATION FROM BOTH STRATEGIES
# (over given range of K values)
kRange = range(2,11)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for i in range(2) :     # Two initializations for each strategy
    
    # K-Means with Strategy 1
    objFuncVal_strat1 = []
    for k in kRange : 
        print("\nIMPLEMENTING KMEANS CLUSTERING WITH RANDOM INITIAL CENTROIDS, K={}\n".format(k))
        objFuncVal_strat1.append(kMeans(k, inpData, 1))

    # K-Means with Strategy 2
    objFuncVal_strat2 = []
    for k in kRange : 
        print("\nIMPLEMENTING KMEANS CLUSTERING WITH KMEANS++ INITIAL CENTROIDS, K={}\n".format(k))
        objFuncVal_strat2.append(kMeans(k, inpData, 2))


    # PLOTTING OBJECTIVE FUNCTION VALUES vs K VALUES GRAPHS

    axs[0][i].plot(kRange, objFuncVal_strat1)
    axs[0][i].set_title("K-Means with Strategy 1, Initialization %s" %(i+1))
    axs[0][i].set_xlabel("K values")
    axs[0][i].set_ylabel("Objective function values")
    axs[0][i].grid()

    axs[1][i].plot(kRange, objFuncVal_strat2)
    axs[1][i].set_title("K-Means with Strategy 2, Initialization %s" %(i+1))
    axs[1][i].set_xlabel("K values")
    axs[1][i].set_ylabel("Objective function values")
    axs[1][i].grid()

for ax in fig.get_axes():
    ax.label_outer()
plt.suptitle("Objective function (Je)   vs   No. of Clusters (k)")
plt.rc('lines', linewidth=3.0)
plt.show()