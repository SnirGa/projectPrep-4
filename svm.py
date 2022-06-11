import pandas as pd
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from os import listdir
from os.path import isfile, join
from sklearn import svm
from sklearn.metrics import make_scorer, auc
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import functions_module
roc_curve_scorer = make_scorer(metrics.roc_curve)
average_precision_scorer = make_scorer(metrics.average_precision_score)
# roc_auc_score_scorer = make_scorer(metrics.roc_auc_score(multi_class="ovr", average="weighted"))
scoring = {"accuracy": "accuracy","precision_micro": "precision_micro","roc_auc_score_ovr": "roc_auc_score_ovr", "roc_curve_ovr": roc_curve_scorer, "average_precision": average_precision_scorer}

def get_x_y_lsts(dataset_path):
    csv_lst = functions_module.read_csv(dataset_path)
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

datasets = ["datasets/"+f for f in listdir("datasets") if isfile(join("datasets", f))]
cv_outer = KFold(n_splits=10, shuffle=True)
outer_results = list()
for data in datasets:
    X, Y = get_x_y_lsts(data)
    # ---- configure the cross-validation procedure ----
    cv_inner = KFold(n_splits=10, shuffle=True, random_state=1)
    # ---- define the SVM model ----
    # Support Vector Machine algorithms are not scale invariant, so it is highly recommended to scale the data.
    # Standardize features by removing the mean and scaling to unit variance.
    model = make_pipeline(StandardScaler(), svm.SVC())
    # ---- define search parmas for hyperparameter optimization ----
    params = dict()
    params['svc__C'] = [1, 1.5, 2, 2.5, 3]
    params['svc__kernel'] = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
    # ---- define search ----
    search = RandomizedSearchCV(model, params, scoring=scoring, n_jobs=1, cv=cv_inner, refit='accuracy', random_state=42)
    # ---- execute the outer nested cross-validation ----
    # scores = cross_validate(search, X, Y, scoring= scoring, cv=cv_outer, n_jobs=-1)
    # ---- Training Time metric ----
    start = time.time()
    # execute search
    result = search.fit(X, Y)
    stop = time.time()
    training_time = stop - start
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    # evaluate model on the hold out dataset
    yhat = best_model.predict(X)
    # evaluate the model
    acc = metrics.accuracy_score(Y, yhat)
    FPR, TPR, _ = metrics.roc_curve(Y, yhat)
    Precision = metrics.precision_score(Y, yhat,average='micro')
    AUC_ROC_Curve = metrics.roc_auc_score(Y, yhat,average='micro')
    precision, recall, _ = metrics.precision_recall_curve(Y, yhat)
    auc_precision_recall = auc(recall, precision)
    # store the result
    outer_results.append(acc)
    outer_results.append(FPR)
    outer_results.append(TPR)
    outer_results.append(Precision)
    outer_results.append(AUC_ROC_Curve)
    outer_results.append(auc_precision_recall)
    outer_results.append(training_time)
    outer_results.append(training_time) #Inference time for 1000 instances
    # report progress
    # print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
# summarize the estimated performance of the model
# print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))



