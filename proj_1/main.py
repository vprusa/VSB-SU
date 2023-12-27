import getopt
import sys
import csv
from pprint import pprint

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def parse_csv_to_2d_array(filename):
    df = pd.read_csv(filename)
    return df

def train_model(features_train, target_train):
    print("train")
    # Train a Classifier
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(features_train, target_train)

    return classifier

def classify_model(classifier, feature_test, target_test):
    print('classify')
    target_pred = classifier.predict(feature_test)
    accuracy = accuracy_score(target_test, target_pred)
    print(f"Model Accuracy: {accuracy}")
    pass

def prepare_data(data, target_variable):
    # Preprocess the Data - Convert categorical variables to numerical
    label_encoders = {}

    print("Data:")
    pprint(list(data.columns))
    print("Target:")
    print(target_variable)
    for column in data.columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # Split the Data to feature and train data
    #featurs_columns = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
    #featurs_columns = data.columns.copy(deep=True)
    featurs_columns = list(data.columns.values)
    featurs_columns.remove(target_variable)
    features_data = data[featurs_columns]  # Features
    target_data = data[target_variable]  # Target variable
    features_train, features_test, target_train, target_test = train_test_split(
        features_data, target_data, test_size=0.2, random_state=42)

    return (features_train, features_test, target_train, target_test)

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

    # Standardize the data
    featurs_columns = list(data.columns.values)
    featurs_columns.remove(target_variable)
    # Clustering using KMeans
    kmeans = KMeans(n_clusters=clusters_cnt, random_state=1)

    # Calculate clusters and visualize them in single plot
    rows_cnt = 1
    cols_cnt = 2
    size = 5
    fig, ax = plt.subplots(rows_cnt, cols_cnt, sharex='col', sharey='row', figsize=(size * cols_cnt, size * rows_cnt))
    n = cols_cnt * rows_cnt
    axes = ax.flatten()
    plot_x = target_variable_x  # 'Annual Income (k$)'
    plot_y = target_variable_y  # 'Spending Score (1-100)'
    plt.xlabel(plot_x)
    plt.ylabel(plot_y)
    plt.title('Clusters of customers')
    for i, j in zip(range(n), axes):
        result_data_name = 'Cluster-' + str(i)
        featurs_columns = list(data.columns.values)

        target_featurs_columns = list(data.columns.values)
        target_featurs_columns.remove(target_variable_x)
        target_featurs_columns.remove(target_variable_y)
        if len(target_featurs_columns) < i:
            continue
        target_variable = target_featurs_columns[i]
        # if target_variable == target_variable_x or target_variable == target_variable_y:
        #     continue
        featurs_columns.remove(target_variable)
        scaler = StandardScaler()
        # scaled_features = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
        scaled_features = scaler.fit_transform(data[featurs_columns])

        data[result_data_name] = kmeans.fit_predict(scaled_features)
        data_sum = sum(data[result_data_name])
        # print("data sum - " + result_data_name + ": " + str(data_sum))
        print("target: " + target_variable + " data sum - " + result_data_name + ": " + str(data_sum))
        j.scatter(data[plot_x], data[plot_y], c=data[result_data_name])
    plt.show()

if __name__ == "__main__":
   main(sys.argv[1:])
