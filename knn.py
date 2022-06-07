from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from numpy import mean
from numpy import std

#read csv:
df = pd.read_csv("C:/Users/ronig/Desktop/לימודים/שנה ג/סימסטר ב/סדנת הכנה לפרויקט בהנדסת/עבודה 4/dataSets/arrhythmia.csv", )
X = df.drop("clase",axis=1).to_numpy()  #Feature Matrix
y = df["clase"].to_numpy()       #Target Variable
# print(X)
# print(y)

#https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/
# configure the cross-validation procedure
cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
# define the model
model = RandomForestClassifier(random_state=1)
# define search space
space = dict()
space['n_estimators'] = [10, 100, 500]
space['max_features'] = [2, 4, 6]
# define search
search = GridSearchCV(model, space, scoring='accuracy', n_jobs=1, cv=cv_inner, refit=True)
# configure the cross-validation procedure
cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
# execute the nested cross-validation
scores = cross_val_score(search, X, y, scoring='accuracy', cv=cv_outer, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

#
# # choose k between 1 to 31
# k_range = range(0, 15)
# k_scores = []
# # use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
#     k_scores.append(scores.mean())
# # plot to see clearly
# plt.plot(k_range, k_scores)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-Validated Accuracy')
# plt.show()
#
# # Use neg_mean_squared_error for scoring(good for regression)
# k_range = range(0, 15)
# k_scores = []
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     loss = abs(cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error'))
#     k_scores.append(loss.mean())
# plt.plot(k_range, k_scores)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-Validated MSE')
# plt.show()
#
#
# #we see that the best point in that range is : 3
# print("-----------------------------------n_neighbors = 3-----------------------------------")
#
# knn = KNeighborsClassifier(n_neighbors = 3)
# # X,y will automatically devided by 10 folder, the scoring I will still use the accuracy
# scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
# # print all 10 times scores
# # print(scores)
# #scores = [0.63043478 0.60869565 0.57777778 0.57777778 0.6 0.55555556 0.57777778 0.57777778 0.62222222 0.64444444]
# # then I will do the average about these 10 scores to get more accuracy score.
# # print(scores.mean())
# #scores.mean() = 0.5972463768115943
#
