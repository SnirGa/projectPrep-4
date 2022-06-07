import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def get_x_y_lsts(dataset_path):
    csv_lst = pd.read_csv(dataset_path)
    X = []
    Y = []
    features_number = len(csv_lst[0]) - 1 #???????
    for row in csv_lst:
        row_features = []
        for i in range(features_number):
            row_features.append(row[i])
        X.append(row_features)
        Y.append(row[-1])
    return X, Y

datasets = ["datasets/"+f for f in listdir("datasets") if isfile(join("datasets", f))]
# print(datasets)
for data in datasets:
    X, Y = get_x_y_lsts(data)
    cv_inner = KFold(n_splits=10, shuffle=True)
    # ---- define the SVM model ----
    # Support Vector Machine algorithms are not scale invariant, so it is highly recommended to scale the data.
    # Standardize features by removing the mean and scaling to unit variance.
    model = make_pipeline(StandardScaler(), svm.SVC())
    # ---- define search parmas for hyperparameter optimization ----
    params = dict()
    params['C'] = [1, 1.5, 2, 2.5, 3]
    params['kernel'] = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
    # ---- Training Time metric ----
    start = time.time()
    # ---- define search ----
    search = RandomizedSearchCV(model, params, scoring=['accuracy','roc_curve','precision','roc_auc_score','average_precision'], n_jobs=1, cv=cv_inner, refit=True)
    # ---- See the highest cross-validated accuracy score and the optimal value for the hyperparameters
    print(search.best_score_)
    print(search.best_params_)
    # ---- configure the cross-validation procedure ----
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    # ---- execute the outer nested cross-validation ----
    scores = cross_val_score(search, X, Y, scoring=['accuracy','roc_curve','precision','roc_auc_score','average_precision'], cv=cv_outer, n_jobs=-1)
    stop = time.time()



