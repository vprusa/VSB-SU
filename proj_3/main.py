# zadani
# implementovat CLARANS
#
# https://homel.vsb.cz/~pla06/
# CLARANS je
# https://homel.vsb.cz/~pla06/files/ml/ml_04.pdf
# Scalable Data Clustering
# CLARA
# • A scalable implementation of the k-medoids algorithm.
# • k-medoids
# • Initial clusters are selected randomly.
# • A set of pairs (X, Y) is randomly generated, X from dataset and Y from
# representatives, and the best pair is used for exchange.
# • Based on the Partitioning Around Medoids (PAM).
# • All possible (k · (n − k)) pairs are evaluated for replacement and the best is
# selected.
# • This is repeated until convergence to the local optimum.
# • O(kn2d)time complexity for d-dimensional dataset.
#
#• Due to high computational complexity only subset of points is selected for
# exhaustive search.
# • A sampling fraction f is selected, where f << 1.
# • Non-sampled points are assigned to the nearest medoids.
# • The sampling is repeated over independently chosen samples os the same
# size f · n.
# • The best clustering is then selected.
# • Time complexity if for one iteration
# O(k · f 2 · n2 · d + k · (n − k))
#
# CLARANS
# Solves problem of CLARA when no good choice of medoids is present in any
# of the sample.
# • CLARANS states for Clustering Large Applications based on Randomized
# Search.
# • The algorithm works with the full data set (no samples).
# • The algorithm iteratively attempts exchanges between random medoids with
# random non-medoids.
#
#The quality of the exchange is checked after each attempts.
# • When improves, the exchange is made final.
# • When not, unsuccessful exchange attempts counter is incremented.
# • A local optimal solution is found when a user-specified number of
# unsuccessful attempts MaxAttempt is reached.
# • This process of finding the local optimum is repeated for a user-specified
# number of iterations - MaxLocal.
# • The clustering objective is evaluated for each iteration and the best is
# selected as the optimal.
# • The advantage of CLARANS over CLARA is that a greater diversity of the
# search space is explored.
#
#
# dalsi odkazy na procteni
# https://analyticsindiamag.com/comprehensive-guide-to-clarans-clustering-algorithm/
# reasearchgate clarabs algorithm
# https://www.researchgate.net/figure/Pseudo-code-for-Pro-CLARANS-algorithm_fig5_24253045
# other lecture
# https://www.dbs.ifi.lmu.de/Lehre/KDD/SS12/skript/kdd-7-Clustering_part_1.pdf

import getopt
import sys
from pprint import pprint

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
import numpy as np

class Clarans(object):

    def compute_cost(self, data, medoids):
        cost = 0
        for point in data:
            cost += np.min([np.linalg.norm(point - medoid) for medoid in medoids])
        return cost

    def get_neighbor(self, medoids, data):
        neighbor = medoids.copy()
        # Choose a medoid to be replaced
        to_replace = random.choice(range(len(medoids)))
        # Choose a non-medoid point as a new medoid
        # non_medoids = [point for point in data if point not in medoids]
        medoids_tuples = [tuple(medoid) for medoid in medoids]
        non_medoids = []
        # Iterate over each point in the data
        for point in data:
            point_tuple = tuple(point)
            # Check if the point is not a medoid
            if point_tuple not in medoids_tuples:
                non_medoids.append(point)

        replacement = random.choice(non_medoids)
        neighbor[to_replace] = replacement
        return neighbor

    def cluster(self, data, num_clusters, maxneighbor, numlocal):
        best_medoids = None
        best_cost = float('inf')

        self.medoids

        for iteration in range(numlocal):
            print("current iteration: ", str(iteration))
            # Randomly select initial medoids
            current_medoids = random.sample(list(data), num_clusters)
            current_cost = self.compute_cost(data, current_medoids)

            num_examinations = 0
            while num_examinations < maxneighbor:
                neighbor_medoids = self.get_neighbor(current_medoids, data)
                neighbor_cost = self.compute_cost(data, neighbor_medoids)

                if neighbor_cost < current_cost:
                    current_medoids = neighbor_medoids
                    current_cost = neighbor_cost
                    num_examinations = 0  # Reset counter
                else:
                    num_examinations += 1

            if current_cost < best_cost:
                best_medoids = current_medoids
                best_cost = current_cost

        return best_medoids, best_cost

def main(argv):
    input_file = ''
    target_variable = ''
    verbose = False

    # Define the usage message
    usage = 'Usage: script.py -i <inputfile> -t <target> -x <x_target>  -y <y_target> -c <clusters> -g <ignore> [-v]'

    try:
        # Parse the command-line arguments
        opts, args = getopt.getopt(argv, "hi:t:x:y:g:c:v", ["ifile=", "target=", "xtarget=", "ytarget=", "clusters=", "ignore="])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    clusters_cnt = 5
    for opt, arg in opts:
        if opt == '-h':
            print(usage)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-t", "--target"):
            target_variable = arg
        elif opt in ("-x", "--xtarget"):
            target_variable_x = arg
        elif opt in ("-y", "--ytarget"):
            target_variable_y = arg
        elif opt in ("-c", "--clusters"):
            clusters_cnt = int(arg)
        elif opt in ("-g", "--ignore"):
            ignore_col = arg
        elif opt == '-v':
            verbose = True

    # Process the input and output files
    print('Input file is "', input_file, '"')
    print('Target variable is "', target_variable, '"')
    if verbose:
        print("Verbose mode is enabled")
    if input_file is None or target_variable is None or target_variable_x is None or target_variable_y is None:
        print(usage)
        exit(1)

    label_encoders = {}

    # Read the CSV Data
    data = pd.read_csv(input_file)

    print("Data:")
    pprint(list(data.columns))
    print("Target:")
    print(target_variable)

    # Convert categorical columns to numerical columns
    for column in data.columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    if ignore_col is not None:
        data = data.drop(ignore_col, axis='columns')

    # Convert categorical data to numerical data
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])

    # Select the columns you want to use in clustering
    # Assuming you want to use all the columns except 'CustomerID'
    # clustering_data = data.drop('CustomerID', axis=1)
    clustering_data = data

    # Optionally, normalize the data
    scaler = StandardScaler()
    clustering_data_normalized = scaler.fit_transform(clustering_data)

    # Convert to NumPy array for the clarans function
    npdata = np.array(clustering_data_normalized)

    # Now you can use the data in the clarans function
    num_clusters = 5
    maxneighbor = 5
    numlocal = 100  # 1000

    # orig_target_variable = target_variable
    # data = np.random.rand(100, 2)  # Generate some 2D points
    # num_clusters = 3
    # maxneighbor = 5
    # numlocal = 4
    #
    clarans = Clarans()
    best_medoids, best_cost = clarans.cluster(npdata, num_clusters, maxneighbor, numlocal)
    # print("Best Medoids:", best_medoids)
    # print("Best Cost:", best_cost)
    # Assuming 'Annual Income (k$)' is the x-axis and 'Spending Score (1-100)' is the y-axis
    x_axis = target_variable_x  # 'Annual Income (k$)'
    y_axis = target_variable_y  # 'Spending Score (1-100)'

    # Reverse the normalization to plot in the original scale
    data_denormalized = scaler.inverse_transform(npdata)

    x_idx = clustering_data.columns.get_loc(x_axis)
    y_idx = clustering_data.columns.get_loc(y_axis)

    # Extract the columns for the x and y axis
    x = data_denormalized[:, x_idx]
    y = data_denormalized[:, y_idx]

    # Plotting all points
    plt.scatter(x, y, c='grey', label='Data points')

    # Highlighting the medoids
    # for medoid in best_medoids:
    #     medoid_denormalized = scaler.inverse_transform([npdata[medoid]])
    #     # plt.scatter(npdata[x_idx],
    #     #             npdata[y_idx],
    #     plt.scatter(medoid_denormalized[:, x_idx],
    #                 medoid_denormalized[:, y_idx],
    #                 c='red',
    #                 label='Medoid' if list(best_medoids).index(medoid) == 0 else "")

    # # Highlighting the medoids
    # for medoid_idx in best_medoids:
    #     medoid = npdata[medoid_idx]
    #     medoid_denormalized = scaler.inverse_transform([medoid])
    #     plt.scatter(medoid_denormalized[:, x_idx],
    #                 medoid_denormalized[:, y_idx],
    #                 c='red',
    #                 label='Medoid' if best_medoids.index(medoid_idx) == 0 else "")
    # plt.scatter(x, y, c='grey', label='Data points')

    # Highlighting the medoids
    for medoid in best_medoids:
        # Find the index of the medoid in npdata
        medoid_idx = np.where(np.all(npdata == medoid, axis=1))[0][0]

        # Use the index to get the data point for denormalizing and plotting
        medoid_denormalized = scaler.inverse_transform([npdata[medoid_idx]])
        plt.scatter(medoid_denormalized[:, x_idx],
                    medoid_denormalized[:, y_idx],
                    c='red',
                    label='Medoid')
        # label = 'Medoid' if best_medoids.index(medoid) == 0 else "")

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title('CLARANS Clustering')
    # plt.legends()
    plt.show()

if __name__ == "__main__":
   main(sys.argv[1:])
