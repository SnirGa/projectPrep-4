from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.naive_bayes import GaussianNB
import numpy as np

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
    kf = KFold(n_splits=10)
    accuracy=[]
    tpr=[]
    fpr=[]
    precision=[]
    auc_roc_curve=[]
    auc_precision_recall=[]
    training_time=[]
    inference_time=[]


    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for train_index, test_index in kf.split(x):
        x_train = [x[i] for i in train_index]
        x_test = [x[i] for i in test_index]
        y_train = [y[i] for i in train_index]
        y_test = [y[i] for i in test_index]
        gnb=get_best_gnb(x,y)
        gnb.fit(x_train,y_train)
        y_prediction=gnb.predict(x_test)




csvs = get_csvs()
for csv in csvs:
    csv_lst = read_csv(csv)
    x, y = get_x_y_lsts(csv)
    get_metrics(x,y)
