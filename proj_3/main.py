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
#
# https://en.wikipedia.org/wiki/K-medoids
#
# Partitioning Around Medoids (PAM)
# PAM[1] uses a greedy search which may not find the optimum solution,
# but it is faster than exhaustive search. It works as follows:
#
#     (BUILD) Initialize: greedily select k of the n data points as the medoids to minimize the cost
#     Associate each data point to the closest medoid.
#     (SWAP) While the cost of the configuration decreases:
#         For each medoid m, and for each non-medoid data point o:
#             Consider the swap of m and o, and compute the cost change
#             If the cost change is the current best, remember this m and o combination
#         Perform the best swap of m best {\displaystyle m_{\text{best}}}
#         and o best {\displaystyle o_{\text{best}}}, if it decreases the cost function.
#         Otherwise, the algorithm terminates.
#
# Other approximate algorithms such as CLARA and CLARANS trade quality for runtime.
# CLARA applies PAM on multiple subsamples, keeping the best result. CLARANS works on the entire data set,
# but only explores a subset of the possible swaps of medoids and non-medoids using sampling.

import getopt
import sys
from pprint import pprint
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# There may be problems running the visualization on different platforms
# One of this options may fix the issues
import matplotlib; matplotlib.use("TkAgg")
# import matplotlib; matplotlib.use("svg")
# ValueError: 'gtkagg' is not a valid value for backend; supported values are
# ['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo',
# 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
# import matplotlib; matplotlib.use("Agg")

# visualization color palette pairs
MY_COLORS = [
    # ('#FFB6C1', '#FF0000'),  # lightred, darkred
    # ('#ADD8E6', '#00008B'),  # lightblue, darkblue
    # ('#90EE90', '#006400'),  # lightgreen, darkgreen
    # ('#F08080', '#8B0000'),  # lightcoral, darkcoral
    # ('#E0FFFF', '#008B8B'),  # lightcyan, darkcyan
    # # ('#FAFAD2', '#BDB76B'),  # lightgoldenrodyellow, darkgoldenrod
    # # ('#D3D3D3', '#A9A9A9'),  # lightgray, darkgray
    # # ('#FFB6C1', '#C71585'),  # lightpink, darkpink
    # # ('#FFA07A', '#FF4500'),  # lightsalmon, darksalmon
    # ('#20B2AA', '#008080'),  # lightseagreen, darkseagreen
    # ('#87CEFA', '#0000CD'),  # lightskyblue, darkskyblue
    ('#FFB6C1', '#FF0000'),  # lightred, darkred
    ('#ADD8E6', '#00008B'),  # lightblue, darkblue
    ('#90EE90', '#006400'),  # lightgreen, darkgreen
    ('#F08080', '#8B0000'),  # lightcoral, darkcoral
    ('#E0FFFF', '#008B8B'),  # lightcyan, darkcyan
    ('#FAFAD2', '#BDB76B'),  # lightgoldenrodyellow, darkgoldenrod
    ('#D3D3D3', '#A9A9A9'),  # lightgray, darkgray
    ('#FFB6C1', '#C71585'),  # lightpink, darkpink
    ('#FFA07A', '#FF4500'),  # lightsalmon, darksalmon
    ('#20B2AA', '#008080'),  # lightseagreen, darkseagreen
    ('#87CEFA', '#0000CD'),  # lightskyblue, darkskyblue
    ('#778899', '#2F4F4F'),  # lightslategray, darkslategray
    ('#B0C4DE', '#4682B4'),  # lightsteelblue, darksteelblue
    ('#FFFFE0', '#FFD700'),  # lightyellow, darkyellow
    ('#FFA07A', '#DC143C'),  # lightcoral, darkcoral
    ('#E0FFFF', '#00CED1'),  # lightcyan, darkturquoise
    ('#98FB98', '#006400'),  # palegreen, darkgreen
    ('#AFEEEE', '#40E0D0'),  # paleturquoise, turquoise
    ('#DB7093', '#C71585'),  # palevioletred, mediumvioletred
    ('#FFDAB9', '#FF8C00')  # peachpuff, darkorange
]

class Clarans(object):
    """
    This class contains support operations for CLARANS

    Originally it was supposed to contain all operations,
    but because of limitations of Visualization using plt.
    The main function `update` was moved as static.
    This code-smell is small price to pay for working visualization.
    """

    def compute_cost(self, data, medoids):
        """
        Computes cost for medoids cluster.
        """

        cost = 0
        for point in data:
            # Initialize the minimum distance to a large number
            min_distance = float('inf')
            for medoid in medoids:
                # Calculate the distance from the current point to the current medoid
                distance = np.linalg.norm(point - medoid)
                # Update min_distance if the current distance is smaller
                if distance < min_distance:
                    min_distance = distance
            # Add the minimum distance to the cost
            cost += min_distance
        return cost

    def get_neighbor(self, medoids, data):
        """
        Get random neighbour for medoids from data.
        """
        neighbor = medoids.copy()
        # pick medoid to be replaced
        to_replace = random.choice(range(len(medoids)))
        # pick non-medoid point as a new medoid
        medoids_tuples = [tuple(medoid) for medoid in medoids]
        non_medoids = []
        # search for new data point
        for point in data:
            point_tuple = tuple(point)
            # Check if the point is not a medoid
            if point_tuple not in medoids_tuples:
                non_medoids.append(point)

        # replace old point with a new one
        replacement = random.choice(non_medoids)
        neighbor[to_replace] = replacement
        return neighbor


def get_color_name(input_integer, light=False):
    """
    Get color name by index.
    """
    # Get a list of color names
    color_index = (input_integer) % len(MY_COLORS)
    # Return the color name
    if light:
        return MY_COLORS[color_index][0]
    else:
        return MY_COLORS[color_index][1]

class MedoidCluster(object):
    """
    MedoidCluster crate for medoid and scatter data.
    """
    def __init__(self, cs, ms, m):
        self.cluster_scatter = cs
        self.medoid_scatter = ms
        self.medoid = m

def main(argv):
    input_file = ''
    verbose = False
    max_neighbours = 5
    max_iterations = 10
    visualize_anim = True
    categorical_col = None
    # Define the usage message
    usage = 'Usage: script.py -i <inputfile> -x <x_target>  -y <y_target> -c <clusters> -g <ignore_cols:i_1,i_2,...,i_n> -c <categorical::i_1,i_2,...,i_n> [-v] [-a]'

    try:
        # Parse the command-line arguments
        opts, args = getopt.getopt(argv, "hi:x:y:g:c:c:n:r:va",
                                   ["ifile=", "xtarget=", "ytarget=",
                                    "ignore=", "categorical=", "clusters=", "neighbours=", "iterations="])
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
            ignore_col = arg.split(",")
        elif opt in ("-g", "--categorical"):
            categorical_col = arg.split(",")
        elif opt == '-v':
            verbose = True
        elif opt == '-a':
            # if visualization should be disabled, for debugging purposes
            visualize_anim = False

    # Process the input and output files
    print('Input file is "', input_file, '"')
    if verbose:
        print("Verbose mode is enabled")
    if input_file is None or target_variable_x is None or target_variable_y is None:
        print(usage)
        exit(1)

    label_encoders = {}

    # Read the CSV Data
    rawdata = pd.read_csv(input_file)

    print("Data:")
    pprint(list(rawdata.columns))
    print("max_iterations:")
    print(max_iterations)

    print("target_variable_x:")
    print(target_variable_x)
    print("target_variable_y:")
    print(target_variable_y)

    # Convert categorical columns to numerical columns
    for column in rawdata.columns:
        label_encoders[column] = LabelEncoder()
        rawdata[column] = label_encoders[column].fit_transform(rawdata[column])

    if ignore_col is not None:
        for i in ignore_col:
            clustering_data = rawdata.drop(i, axis='columns')

    # Convert categorical data to numerical data, normalize the data
    if categorical_col is not None:
        label_encoder = LabelEncoder()
        for i in categorical_col:
            clustering_data[i] = label_encoder.fit_transform(clustering_data[i])
    scaler = StandardScaler()
    clustering_data_normalized = scaler.fit_transform(clustering_data)

    # Convert to NumPy array for the clarans function
    data = np.array(clustering_data_normalized)

    # Reverse the normalization to plot in the original scale
    data_denormalized = scaler.inverse_transform(data)

    # get plot data columns indexes
    x_idx = clustering_data.columns.get_loc(target_variable_x)  # 'Annual Income (k$)'
    y_idx = clustering_data.columns.get_loc(target_variable_y)  # 'Spending Score (1-100)'

    # Extract the columns for the x and y axis
    x_data = data_denormalized[:, x_idx]
    y_data = data_denormalized[:, y_idx]

    clarans = Clarans()
    Clarans.best_medoids = None
    Clarans.best_cost = float('inf')

    fig, ax = plt.subplots()
    plt.title("CLARANS")
    plt.xlabel(target_variable_x)
    plt.ylabel(target_variable_y)

    Clarans.medoid_scatters = []
    ax.scatter(x_data, y_data, c='lightgrey', label='Data points')
    for i in range(clusters_cnt):
        mc = MedoidCluster(ax.scatter([], [], c=get_color_name(i, True), marker='.', label='Medoids'),
                           ax.scatter([], [], c=get_color_name(i, False), marker='X', label='Medoids'),
                           None)
        Clarans.medoid_scatters.append(mc)
    # medoid_scatter = Clarans.medoid_scatters

    Clarans.max_iteration = max_iterations
    Clarans.iteration = 0

    def assign_points_to_medoids(pdata, medoids):
        # Create a list of empty lists to store points for each medoid
        clusters = [[] for _ in medoids]
        for point in pdata:
            # Compute distances from the current point to each medoid
            distances = [np.linalg.norm(point - medoid) for medoid in medoids]
            # Find the index of the closest medoid
            closest_medoid_idx = np.argmin(distances)
            # Assign the point to the cluster of the closest medoid
            clusters[closest_medoid_idx].append(point)
        return clusters
    Clarans.best_medoids = None
    Clarans.best_cost = float('inf')
    Clarans.current_medoids = random.sample(list(data), clusters_cnt)
    Clarans.current_cost = clarans.compute_cost(data, Clarans.current_medoids)

    def update(frame):
        print("current iteration: ", str(Clarans.iteration))

        num_examinations = 0
        while num_examinations < max_neighbours:

            # calculate new neighbour cost
            neighbor_medoids = clarans.get_neighbor(Clarans.current_medoids, data)
            neighbor_cost = clarans.compute_cost(data, neighbor_medoids)

            # if the cost is better, then continue search for minimal cost
            if neighbor_cost < Clarans.current_cost:
                Clarans.current_medoids = neighbor_medoids
                Clarans.current_cost = neighbor_cost
                num_examinations = 0  # Reset counter
            else:
                num_examinations += 1

        # if new cost is better, then update next medoids
        if Clarans.current_cost < Clarans.best_cost:
            Clarans.best_medoids = Clarans.current_medoids
            Clarans.best_cost = Clarans.current_cost

        # Update plot with medoids and cluster
        #   Assign points to medoids and get clusters
        clusters = assign_points_to_medoids(data, Clarans.best_medoids)
        #   Update plot for each medoid and its cluster
        for idx, cluster in enumerate(clusters):
            cluster_points = np.array(cluster)
            if len(cluster_points) > 0:
                cluster_data_denormalized = scaler.inverse_transform(cluster_points)
                cluster_plot_data = list(zip(list(cluster_data_denormalized[:, x_idx]), list(cluster_data_denormalized[:, y_idx])))
                Clarans.medoid_scatters[idx].cluster_scatter.set_offsets(cluster_plot_data)
            # Update medoid scatter plot
            medoid_point = np.array([Clarans.best_medoids[idx]])
            medoid_point_denormalized = scaler.inverse_transform(medoid_point)
            Clarans.medoid_scatters[idx].medoid_scatter.set_offsets(medoid_point_denormalized[:, [x_idx, y_idx]])

        Clarans.iteration = Clarans.iteration + 1
        # It may be good to mention that the PAM terminates when the Cost is not decreasing anymore
        # That is not possible here, because of the randomness of choosing next sample
        # Some heuristic for termination may improve the run time
        if Clarans.iteration > Clarans.max_iteration:
            # because the FuncAnimation does not terminate properly, force it here
            exit(0)


    sleep_time = 0.25
    if visualize_anim:
        ani = FuncAnimation(fig=fig, func=update, frames=max_iterations, interval=sleep_time*1000)
        plt.show()
    else:
        plt.show()
        # For debug purposes because the pycharm is not compatible with FuncAnimation debugging
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
