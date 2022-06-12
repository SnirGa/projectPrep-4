import time
import statistics
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn import metrics

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier


def get_mlp_model(hiddenLayerOne=784, hiddenLayerTwo=256, dropout=0.2, learnRate=0.01):
    # initialize a sequential model and add layer to flatten the input data
    model = Sequential()
    model.add(Flatten())
    # add two stacks of FC => RELU => DROPOUT
    model.add(Dense(hiddenLayerOne, activation="relu", input_shape=(784,)))
    model.add(Dropout(dropout))
    model.add(Dense(hiddenLayerTwo, activation="relu"))
    model.add(Dropout(dropout))
    # add a softmax layer on top
    model.add(Dense(10, activation="softmax"))
    # compile the model
    model.compile(optimizer=Adam(learning_rate=learnRate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def define_grid_param():
    # define a grid of the hyperparameter search space
    hiddenLayerOne = [256, 512, 784]
    hiddenLayerTwo = [128, 256, 512]
    learnRate = [1e-2, 1e-3, 1e-4]
    dropout = [0.3, 0.4, 0.5]
    batchSize = [4, 8, 16, 32]
    epochs = [10, 20, 30, 40]
    # create a dictionary from the hyperparameter grid
    grid = dict(
        hiddenLayerOne=hiddenLayerOne,
        learnRate=learnRate,
        hiddenLayerTwo=hiddenLayerTwo,
        dropout=dropout,
        batch_size=batchSize,
        epochs=epochs
    )
    return grid


def get_best_model_with_hyper_parameters(x_train, y_train):
    # wrap our model into a scikit-learn compatible classifier
    model = KerasClassifier(build_fn=get_mlp_model, verbose=0)
    # get a defined grid of the hyperparameter search space
    grid = define_grid_param()
    # initialize a random search with a 3-fold cross-validation and then start the hyperparameter search process
    searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=3, param_distributions=grid, scoring="accuracy")
    results = searcher.fit(x_train, y_train)
    return results.best_params_, results.best_estimator_


def get_metrics(x, y):
    accuracy_lst = []
    tpr_lst = []
    fpr_lst = []
    precision_lst = []
    auc_roc_curve_lst = []
    auc_precision_recall_lst = []
    training_time_lst = []
    inference_time_lst = []
    hyper_parameters_lst = []
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(x):
        # Split data
        x_train, x_test = x[train_index],  x[test_index]
        y_train, y_test = y[train_index],  y[test_index]
        # Get optimized hyper parameters model after 3-Fold cross validation
        hyper_parameters, best_modal = get_best_model_with_hyper_parameters(x_train, y_train)
        hyper_parameters_lst.append(hyper_parameters)
        # Train Model and record time
        start_time = time.time()
        best_modal.fit(x_train, y_train)
        finish_time = time.time()
        # Calculate training time
        training_time_lst.append(finish_time - start_time)
        # Get test model classification prediction
        y_prediction = best_modal.predict(x_test)
        # Calculate accuracy
        accuracy_lst.append(metrics.accuracy_score(y_test, y_prediction))
        # Calculate precision
        precision_lst.append(metrics.precision_score(y_test, y_prediction))
        # Calculate TPR,FPR
        conf = confusion_matrix(y_test, y_prediction)
        tn = conf[0][0]
        fn = conf[1][0]
        tp = conf[1][1]
        fp = conf[0][1]
        tpr_lst.append(tp / (tp + fn))
        fpr_lst.append(fp / (fp + tn))
        # Calculate AUC Precision-Recall
        fpr, tpr, thresh = metrics.roc_curve(y_test, y_prediction)
        auc_precision_recall = metrics.auc(fpr, tpr)
        auc_precision_recall_lst.append(auc_precision_recall)
        # Calculate AUC ROC Curve
        auc_roc_curve = metrics.roc_auc_score(y_test, y_prediction)
        auc_roc_curve_lst.append(auc_roc_curve)
        # Calculate inference time
        arr = []
        while len(arr) < 1000:
            arr.append(x_test[0])
        inference_X_input = np.array(arr)
        start_infer_time = time.time()
        best_modal.predict(inference_X_input)
        finish_infer_time = time.time()
        inference_time_lst.append(finish_infer_time - start_infer_time)
    return accuracy_lst, tpr_lst, fpr_lst, precision_lst, auc_roc_curve_lst, auc_precision_recall_lst, training_time_lst, inference_time_lst, hyper_parameters_lst


def print_results(accuracy_lst, tpr_lst, fpr_lst, precision_lst, auc_roc_curve_lst, auc_precision_recall_lst, training_time_lst, inference_time_lst, hyper_parameters_lst):
    print("----Chosen Hyper Parameters----")
    i = 1
    for hyper_parameters in hyper_parameters_lst:
        print("split {} hyper parameters: {}".format(i, hyper_parameters))
        i += 1
    print("###################################################################")
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


def get_x_y_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop("clase", axis=1).to_numpy()  # Feature Matrix
    Y = df["clase"].to_numpy()  # Target Variable
    return X, Y


def get_ann_results(csv_path):
    print('-----' + csv_path + '-----')
    x, y = get_x_y_from_csv(csv_path)
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x = x.astype("float32")
    accuracy_lst, tpr_lst, fpr_lst, precision_lst, auc_roc_curve_lst, auc_precision_recall_lst, training_time_lst, inference_time_lst, hyper_parameters_lst = get_metrics(x, y)
    print_results(accuracy_lst, tpr_lst, fpr_lst, precision_lst, auc_roc_curve_lst, auc_precision_recall_lst, training_time_lst, inference_time_lst, hyper_parameters_lst)
    return accuracy_lst, tpr_lst, fpr_lst, precision_lst, auc_roc_curve_lst, auc_precision_recall_lst, training_time_lst, inference_time_lst
