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
#
# https://en.wikipedia.org/wiki/K-medoids
# Other approximate algorithms such as CLARA and CLARANS trade quality for runtime.
# CLARA applies PAM on multiple subsamples, keeping the best result. CLARANS works on the entire data set,
# but only explores a subset of the possible swaps of medoids and non-medoids using sampling.

import getopt
import sys
from pprint import pprint
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# import matplotlib; matplotlib.use("TkAgg")
# import matplotlib; matplotlib.use("svg")
# ValueError: 'gtkagg' is not a valid value for backend; supported values are
# ['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo',
# 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
# import matplotlib; matplotlib.use("Agg")

MY_CSS4_COLORS = {
    # 'red': '#FF0000',
    'lightred': '#FFB6C1',
    'darkred': '#FF0000',
    'lightblue': '#ADD8E6',
    'darkblue': '#',
    # 'lightcoral': '#F08080',
    'lightcyan': '#E0FFFF',
    'darkcyan': '#E0FFFF',
    # 'lightgoldenrodyellow': '#FAFAD2',
    # 'lightgray': '#D3D3D3',
    'lightgreen': '#90EE90',
    'darkgreen': '#90EE90',
    # 'lightgrey': '#D3D3D3',
    # 'lightpink': '#FFB6C1',
    # 'lightsalmon': '#FFA07A',
    # 'lightseagreen': '#20B2AA',
    # 'lightskyblue': '#87CEFA',
    # 'lightslategray': '#778899',
    # 'lightslategrey': '#778899',
    # 'lightsteelblue': '#B0C4DE',
    # 'lightyellow': '#FFFFE0',
    # 'darkyellow': '#FFFFE0'

}

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

def main(argv):
    input_file = ''
    target_variable = ''
    verbose = False
    max_neighbours = 5
    max_iterations = 10
    # Define the usage message
    usage = 'Usage: script.py -i <inputfile> -t <target> -x <x_target>  -y <y_target> -c <clusters> -g <ignore> [-v]'

    try:
        # Parse the command-line arguments
        opts, args = getopt.getopt(argv, "hi:t:x:y:g:c:n:r:v",
                                   ["ifile=", "target=", "xtarget=", "ytarget=",
                                    "ignore=", "clusters=", "neighbours=", "iterations="])
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
        elif opt in ("-n", "--neighbours"):
            max_neighbours = int(arg)
        elif opt in ("-r", "--iterations"):
            max_iterations = int(arg)
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
    num_clusters = clusters_cnt
    maxneighbor = max_neighbours
    # numlocal = 100 #  10  # 100  # 1000
    numlocal = max_iterations  #  10  # 100  # 1000

    # orig_target_variable = target_variable
    # data = np.random.rand(100, 2)  # Generate some 2D points
    # num_clusters = 3
    # maxneighbor = 5
    # numlocal = 4
    #
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

    clarans = Clarans()
    best_medoids = None
    best_cost = float('inf')

    data_bkp = data
    data = npdata
    best_medoids = None
    best_cost = float('inf')
    medoids_over_iterations = []  # Store medoids of each iteration for the animation

    import matplotlib.colors as mcolors

    def get_color_name(input_integer, light = False):
        # Get a list of color names
        color_names = list(filter(lambda x: "light" in x, list(MY_CSS4_COLORS.keys())))
        # color_names = list(filter(lambda x: "light" in x, list(mcolors.XKCD_COLORS.keys())))
        # color_names = list(mcolors.XKCD_COLORS.keys())
        # color_names = list(mcolors.TABLEAU_COLORS.keys())

        # Calculate the index in the color_names list
        # color_index = ((input_integer * 5) + 20) % len(color_names)
        color_index = (input_integer + 8) % len(color_names)

        # Return the color name
        if light:
            cn = color_names[color_index].replace("light", "dark")
            return cn
        else:
            cn = color_names[color_index].replace("light", "")
            return cn

    class MedoidCluster(object):
        def __init__(self, cs, ms, m):
            self.cluster_scatter = cs
            self.medoid_scatter = ms
            self.medoid = m
        # scatter = None
        # medoid = None

    fig, ax = plt.subplots()
    Clarans.medoid_scatters = []
    ax.scatter(x, y, c='lightgrey', label='Data points')
    for i in range(num_clusters):
        mc = MedoidCluster(ax.scatter([], [], c=get_color_name(i, True), label='Medoids'),
                           ax.scatter([], [], c=get_color_name(i, True), marker='x', label='Medoids'),
                           None)
        Clarans.medoid_scatters.append(mc)
    # medoid_scatter = Clarans.medoid_scatters

    Clarans.iteration = 0

    def assign_points_to_medoids(data, medoids):
        # Create a list of empty lists to store points for each medoid
        clusters = [[] for _ in medoids]
        for point in data:
            # Compute distances from the current point to each medoid
            distances = [np.linalg.norm(point - medoid) for medoid in medoids]
            # Find the index of the closest medoid
            closest_medoid_idx = np.argmin(distances)
            # Assign the point to the cluster of the closest medoid
            clusters[closest_medoid_idx].append(point)
        return clusters
    Clarans.best_medoids = None
    Clarans.best_cost = float('inf')
    # for iteration in range(numlocal):
    def update(frame):
        current_medoids = random.sample(list(data), num_clusters)
        current_cost = clarans.compute_cost(data, current_medoids)
        print("current iteration: ", str(Clarans.iteration))

        num_examinations = 0
        while num_examinations < maxneighbor:
            neighbor_medoids = clarans.get_neighbor(current_medoids, data)
            neighbor_cost = clarans.compute_cost(data, neighbor_medoids)

            if neighbor_cost < current_cost:
                current_medoids = neighbor_medoids
                current_cost = neighbor_cost
                num_examinations = 0  # Reset counter
            else:
                num_examinations += 1

        if current_cost < Clarans.best_cost:
            Clarans.best_medoids = current_medoids
            Clarans.best_cost = current_cost

        # Update plot
        medoids_to_plot = []
        # if Clarans.best_medoids is not None:
        #     idx = 0
        #     for medoid in Clarans.best_medoids:
        #         # Find the index of the medoid in npdata
        #         medoid_idx = np.where(np.all(npdata == medoid, axis=1))[0][0]
        #         # Use the index to get the data point for denormalizing and plotting
        #         medoid_denormalized = scaler.inverse_transform([npdata[medoid_idx]])
        #         medoid_to_plot = list(zip(list(medoid_denormalized[:, x_idx]), list(medoid_denormalized[:, y_idx])))
        #         Clarans.medoid_scatters[idx].medoid_scatter.set_offsets(medoid_to_plot)
        #
        #         # TODO scatter medoid's cluster
        #         # Clarans.medoid_scatters[idx].cluster_scatter.set_offsets()
        #         idx = idx + 1

        # Assign points to medoids and get clusters
        clusters = assign_points_to_medoids(data, Clarans.best_medoids)
        # Update plot for each medoid and its cluster
        for idx, cluster in enumerate(clusters):
            cluster_points = np.array(cluster)
            if len(cluster_points) > 0:
                Clarans.medoid_scatters[idx].cluster_scatter.set_offsets(cluster_points[:, [x_idx, y_idx]])
            # Update medoid scatter plot
            medoid_point = np.array([Clarans.best_medoids[idx]])
            medoid_point_denormalized = scaler.inverse_transform(medoid_point)
            Clarans.medoid_scatters[idx].medoid_scatter.set_offsets(medoid_point_denormalized[:, [x_idx, y_idx]])

        Clarans.iteration = Clarans.iteration + 1


    plt.show()
    # anim = True
    sleep_time = 0.5
    anim = False
    if anim:
        ani = FuncAnimation(fig=fig, func=update, frames=max_iterations, interval=sleep_time*1000)
    else:
        # for debug purposes because the pycharm is not compatible with FuncAnimation debugging
        for i in range(max_iterations):
            update(None)
            # fig.canvas.draw()
            fig.suptitle("Iteration" + str(i))
            fig.show()
            plt.pause(sleep_time)
            plt.clf()
    plt.show()

    plt.ioff()  # Turn off interactive mode

if __name__ == "__main__":
   main(sys.argv[1:])
