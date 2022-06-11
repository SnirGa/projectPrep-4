import pandas as pd
import os

def read_csv(csv_path):
    csv_file = pd.read_csv(csv_path, sep=",")
    csv_lst = csv_file.values.tolist()
    return csv_lst


def get_x_y_lsts(dataset_path):
    csv_lst = read_csv(dataset_path)
    X = []
    Y = []
    features_number = len(csv_lst[0]) - 1
    for row in csv_lst:
        row_features = []
        for i in range(features_number):
            row_features.append(row[i])
        X.append(row_features)
        Y.append(row[-1])
    return X, Y

def get_csvs():
    return ['datasets/'+x for x in os.listdir('datasets')]

def get_XY_from_csv(csvPath):
    df = pd.read_csv(csvPath, sep=",")
    X = df.drop("clase", axis=1).to_numpy()  # Feature Matrix
    y = df["clase"].to_numpy()  # Target Variable
    return X, y
