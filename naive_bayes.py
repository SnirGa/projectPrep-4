import time

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn import metrics
import statistics
from functions_module import get_csvs, read_csv, get_x_y_lsts


def get_gnb_hyper_parameters(x, y):
    gnb = GaussianNB()
    params_NB = {'var_smoothing': np.logspace(0, -9, num=100)}
    rand_search = RandomizedSearchCV(gnb, params_NB, cv=3)
    results = rand_search.fit(x, y)
    return results.best_params_


def get_best_gnb(x, y):
    hyper_parameters = get_gnb_hyper_parameters(x, y)
    gnb = GaussianNB(var_smoothing=hyper_parameters.get("var_smoothing"))
    return gnb


def get_metrics(x, y):
    kf = KFold(n_splits=10,shuffle=True)
    accuracy_lst = []
    tpr_lst = []
    fpr_lst = []
    precision_lst = []
    auc_roc_curve_lst = []
    auc_precision_recall_lst = []
    training_time_lst = []
    inference_time_lst = []

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for train_index, test_index in kf.split(x):
        x_train = [x[i] for i in train_index]
        x_test = [x[i] for i in test_index]
        y_train = [y[i] for i in train_index]
        y_test = [y[i] for i in test_index]
        gnb = get_best_gnb(x, y)
        before_train = time.time()
        gnb.fit(x_train, y_train)
        after_train = time.time()
        y_prediction = gnb.predict(x_test)
        # Calculate accuracy
        accuracy = metrics.accuracy_score(y_test, y_prediction)
        accuracy_lst.append(accuracy)
        # Calculate precision
        precision = metrics.precision_score(y_test, y_prediction)
        precision_lst.append(precision)
        # Calculate TPR,FPR
        conf = confusion_matrix(y_test, y_prediction)
        tn = conf[0][0]
        fn = conf[1][0]
        tp = conf[1][1]
        fp = conf[0][1]
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tpr_lst.append(tpr)
        fpr_lst.append(fpr)


        # Calculate AUC Precision-Recall
        fpr, tpr, thresh = metrics.roc_curve(y_test, y_prediction)
        auc_precision_recall = metrics.auc(fpr, tpr)
        auc_precision_recall_lst.append(auc_precision_recall)
        # Calculate AUC ROC Curve
        auc_roc_curve = metrics.roc_auc_score(y_test, y_prediction)
        auc_roc_curve_lst.append(auc_roc_curve)
        # Calculate training time
        training_time_lst.append(after_train-before_train)
        # Calculate inference time
        inference_instances=[]
        while len(inference_instances)<1000:
            inference_instances.append(x_test[0])
        before_infer=time.time()
        gnb.predict(inference_instances)
        after_infer=time.time()
        inference_time_lst.append(after_infer-before_infer)
    return accuracy_lst,tpr_lst,fpr_lst,precision_lst,auc_roc_curve_lst,auc_precision_recall_lst,training_time_lst,inference_time_lst




def get_gnb_results(csv):
        x, y = get_x_y_lsts(csv)
        accuracy_lst,tpr_lst,fpr_lst,precision_lst,auc_roc_curve_lst,auc_precision_recall_lst,training_time_lst,inference_time_lst=get_metrics(x,y)
        return accuracy_lst,tpr_lst,fpr_lst,precision_lst,auc_roc_curve_lst,auc_precision_recall_lst,training_time_lst,inference_time_lst
def print_gnb_results(csv):
    # csvs = ["breast-cancer", "bank", "acute-nephritis", "acute-inflammation", "blood", "breast-cancer-wisc",
    #     "breast-cancer-wisc-diag", "congressional-voting", "chess-krvkp", "breast-cancer-wisc-prog"]
    # csvs = ['all_datasets/' + csv + '.csv' for csv in csvs]
    # for csv in csvs:
        print('-----------------------'+csv+'-----------------------')
        accuracy_lst,tpr_lst,fpr_lst,precision_lst,auc_roc_curve_lst,auc_precision_recall_lst,training_time_lst,inference_time_lst=get_gnb_results(csv)
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
