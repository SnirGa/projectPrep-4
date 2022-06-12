import statistics

from sklearn.metrics import confusion_matrix
from os import listdir
from os.path import isfile, join
from sklearn.neighbors import (KNeighborsClassifier)
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn import metrics
import time

#first,before the start of learning process, we will define the hyperparameter of the model:
#We will use 2 hyperparamters: n-neighbors and weights
from functions_module import get_XY_from_csv


def get_knn_hyper_parameters(x, y):
    knn = KNeighborsClassifier()
    k_range = list(range(1, 31))
    weight_options = ['uniform', 'distance']
    # create a parameter grid: map the parameter names to the values that should be searched
    # dictionary = dict(key=values, key=values)
    param_grid_knn = dict(n_neighbors=k_range, weights=weight_options)
    rand_search = RandomizedSearchCV(knn, param_grid_knn, cv=3)
    results = rand_search.fit(x, y)
    return results.best_params_

def get_best_knn(x, y):
    hyper_parameters = get_knn_hyper_parameters(x, y)
    n_neighbors = hyper_parameters.get("n_neighbors")
    weights = hyper_parameters.get("weights")
    best_knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    return best_knn

def get_metrics(x, y):
    kf = KFold(n_splits=10,shuffle=True)
    accuracy_lst = []
    precision_lst = []
    tpr_lst = []
    fpr_lst = []
    auc_precision_recall_lst = []
    auc_roc_curve_lst = []
    training_time_lst = []
    inference_time_lst = []
    tpr_lst_to_check = []
    fpr_lst_to_check = []

    for train_index, test_index in kf.split(x):
        x_train = [x[i] for i in train_index]
        x_test = [x[i] for i in test_index]
        y_train = [y[i] for i in train_index]
        y_test = [y[i] for i in test_index]
        best_knn = get_best_knn(x_train, y_train)
        before_train = time.time()
        best_knn.fit(x_train,y_train)
        after_train = time.time()
        y_prediction = best_knn.predict(x_test)
        #calc metrics :
        #Accuracy:
        accuracy = metrics.accuracy_score(y_test, y_prediction)
        accuracy_lst.append(accuracy)
        # Calculate precision
        precision = metrics.precision_score(y_test, y_prediction)
        precision_lst.append(precision)
        #TPR,FPR:
        conf = confusion_matrix(y_test, y_prediction)
        tn = conf[0][0]
        fn = conf[1][0]
        tp = conf[1][1]
        fp = conf[0][1]
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tpr_lst.append(tpr)
        fpr_lst.append(fpr)
        #AUC:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prediction)
        tpr_lst_to_check.append(tpr.tolist())
        fpr_lst_to_check.append(fpr.tolist())
            #AUC Precision-Recall
        auc_precision_recall = metrics.auc(fpr, tpr)
        auc_precision_recall_lst.append(auc_precision_recall)
            #AUC ROC Curve
        auc_roc_curve = metrics.roc_auc_score(y_test, y_prediction)
        auc_roc_curve_lst.append(auc_roc_curve)
        #Training time
        training_time_lst.append(after_train-before_train)
        #Inference time_1000_instances
        inference_instances = []
        while len(inference_instances)<1000:
            inference_instances.append(x_test[0])
        before_infer=time.time()
        best_knn.predict(inference_instances)
        after_infer=time.time()
        inference_time_lst.append(after_infer-before_infer)
    return accuracy_lst,tpr_lst,fpr_lst,precision_lst,auc_roc_curve_lst,auc_precision_recall_lst,training_time_lst,inference_time_lst


def print_knn_results():
    datasets = ["datasets/" + f for f in listdir("datasets") if isfile(join("datasets", f))]
    for data_csv in datasets :
        print('-----------------------' + data_csv + '-----------------------')
        x, y = get_XY_from_csv(data_csv)
        accuracy_lst, tpr_lst, fpr_lst, precision_lst, auc_roc_curve_lst, auc_precision_recall_lst, training_time_lst, inference_time_lst = get_metrics(x, y)
        print("----Accuracy Results----")
        print(accuracy_lst)
        print("----Accuracy Mean----")
        print(statistics.mean(accuracy_lst))
        print("----Accuracy std----")
        print(statistics.stdev(accuracy_lst))
        print("###################################################################")
        print("----TPR Results----")
        print(tpr_lst)
        print("----TPR Mean----")
        print(statistics.mean(tpr_lst))
        print("----TPR std----")
        print(statistics.stdev(tpr_lst))
        print("###################################################################")
        print("----FPR Results----")
        print(fpr_lst)
        print("----FPR Mean----")
        print(statistics.mean(fpr_lst))
        print("----FPR std----")
        print(statistics.stdev(fpr_lst))
        print("###################################################################")
        print("----AUC ROC Curve Results----")
        print(auc_roc_curve_lst)
        print("-----AUC ROC Curve  Mean----")
        print(statistics.mean(auc_roc_curve_lst))
        print("-----AUC ROC Curve  std----")
        print(statistics.stdev(auc_roc_curve_lst))
        print("###################################################################")
        print("----AUC Precision-Recall Results----")
        print(auc_precision_recall_lst)
        print("----AUC Precision-Recall Mean----")
        print(statistics.mean(auc_precision_recall_lst))
        print("----AUC Precision-Recall std----")
        print(statistics.stdev(auc_precision_recall_lst))
        print("###################################################################")
        print("----Training Time Results----")
        print(training_time_lst)
        print("----Training Time Mean----")
        print(statistics.mean(training_time_lst))
        print("----Training Time std----")
        print(statistics.stdev(training_time_lst))
        print("###################################################################")
        print("----Inference Time Results----")
        print(inference_time_lst)
        print("----Inference Time Mean----")
        print(statistics.mean(inference_time_lst))
        print("----Inference Time std----")
        print(statistics.stdev(inference_time_lst))
        print("###################################################################")
