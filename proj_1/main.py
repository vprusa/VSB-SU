import getopt
import sys
from pprint import pprint

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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

    orig_target_variable = target_variable
    if target_variable != "all":
        # Standardize the data
        featurs_columns = list(data.columns.values)
        featurs_columns.remove(target_variable)
        scaler = StandardScaler()
        # scaled_features = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
        scaled_features = scaler.fit_transform(data[featurs_columns])

        # Clustering using KMeans
        kmeans = KMeans(n_clusters=clusters_cnt, random_state=1)

        for i in list(data.columns.values):
            result_data_name = 'Cluster-' + i
            data[result_data_name] = kmeans.fit_predict(scaled_features)

        result_data_name = "Cluster"
        data[result_data_name] = kmeans.fit_predict(scaled_features)
        # # Visualize clusters
        data_sum = sum(data[result_data_name])
        print("data sum - " + result_data_name + ": " + str(data_sum))
        plot_x = target_variable_x  # 'Annual Income (k$)'
        plot_y = target_variable_y  # 'Spending Score (1-100)'
        plt.xlabel(plot_x)
        plt.ylabel(plot_y)
        plt.title('Clustered by: ' + target_variable + ' to ' + str(clusters_cnt) + " clusters")
        plt.scatter(data[plot_x], data[plot_y], c=data[result_data_name])
        plt.savefig('fig_'+input_file + "_" + target_variable + '.png')
        plt.show()

    else:
        # Calculate clusters and visualize them in single plot
        rows_cnt = clusters_cnt
        cols_cnt = 2
        size = 2
        fig, ax = plt.subplots(rows_cnt, cols_cnt, sharex='col', sharey='row', figsize=(size * cols_cnt, size * rows_cnt))
        n = cols_cnt * rows_cnt
        axes = ax.flatten()
        plot_x = target_variable_x  # 'Annual Income (k$)'
        plot_y = target_variable_y  # 'Spending Score (1-100)'
        plt.xlabel(plot_x)
        plt.ylabel(plot_y)
        plt.title('Clusters of customers')

        target_features_columns = list(data.columns.values)
        target_features_columns.remove(target_variable_x)
        target_features_columns.remove(target_variable_y)

        clusters = {}
        for i, j in zip(range(n), axes):
            # Clustering using KMeans to c clusters
            c = int(i / cols_cnt) + 1
            kmeans = KMeans(n_clusters=c, random_state=1)

            result_data_name = 'Cl-' + target_variable + "-" + str(c) + '-' + str(i)
            featurs_columns = list(data.columns.values)

            idx = (i % cols_cnt)
            if len(target_features_columns) < idx:
                continue
            # prepare data
            target_variable = target_features_columns[idx]
            featurs_columns.remove(target_variable)
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(data[featurs_columns])
            # create clusters
            clusters[result_data_name] = kmeans.fit_predict(scaled_features)

            if verbose:
                # sum is just for debugging to distinguish different clusters
                data_sum = sum(clusters[result_data_name])
                print("target: " + target_variable + " data sum - " + result_data_name + ": " + str(data_sum))
            # j.set_title(target_variable
            j.set_title(result_data_name)
            j.scatter(data[plot_x], data[plot_y], c=clusters[result_data_name], marker='.')

        plt.savefig('fig_'+input_file + "_" + orig_target_variable + '.png')
        plt.show()

if __name__ == "__main__":
   main(sys.argv[1:])
