from sklearn import naive_bayes
import pandas as pd
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn import metrics
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


def get_cross_val_naive_bayes(X, Y, cv,metric):
    gnb = naive_bayes.GaussianNB()
    scores = cross_val_score(gnb, X, Y, cv=cv,scoring=metric)
    return scores

#Accuracy->
# --------main----------
#print(read_csv("datasets/arrhythmia.csv"))
X,Y=get_x_y_lsts("datasets/arrhythmia.csv")
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

print(get_cross_val_naive_bayes(X,Y,cv,'accuracy'))
print(metrics.SCORERS.keys())