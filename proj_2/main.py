import getopt
import sys
import csv
from pprint import pprint

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
    features_train, features_test, target_train, target_test = train_test_split(features_data, target_data, test_size=0.2, random_state=42)

    return (features_train, features_test, target_train, target_test)

def main(argv):
    # Define the command-line options and default values
    input_file = ''
    target_variable = ''
    verbose = False

    # Define the usage message
    usage = 'Usage: script.py -i <inputfile> -t <target> [-v]'

    try:
        # Parse the command-line arguments
        opts, args = getopt.getopt(argv, "hi:t:v", ["ifile=", "target="])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(usage)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-t", "--target"):
            target_variable = arg
        elif opt == '-v':
            verbose = True

    # Process the input and output files
    print('Input file is "', input_file, '"')
    print('Target variable is "', target_variable, '"')
    if verbose:
        print("Verbose mode is enabled")
    if input_file is not None:
        data = parse_csv_to_2d_array(input_file)
        features_train, features_test, target_train, target_test = prepare_data(data, target_variable)
        model = train_model(features_train, target_train)
        res = classify_model(model, features_test, target_test)

if __name__ == "__main__":
   main(sys.argv[1:])
