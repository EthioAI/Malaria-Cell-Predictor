import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from skimage import color
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import multiprocessing


PROCESS_MANAGER = multiprocessing.Manager()

def dataset_loader():
    """
    Loads a dataset.
    """
    if not os.path.exists("data"):
        raise Exception("No data folder found!\n Please download the dataset, unzip it and put it in the data folder.")

    X = []
    y = []

    for i in os.listdir("data/cell_images/Parasitized"):
        if ".png" in i:
            path = "data/cell_images/Parasitized/"+i 
            img = cv2.resize(plt.imread(path), (25,25))
            img = color.rgb2gray(img).ravel()
            X.append(img)
            y.append(1)

    for i in os.listdir("data/cell_images/Uninfected/"):
        if ".png" in i:
            path = "data/cell_images/Uninfected/"+i
            img = cv2.resize(plt.imread(path), (25,25))
            img = color.rgb2gray(img).ravel()
            X.append(img)
            y.append(0)

    X = np.array(X) 
    y = np.array(y)

    return X, y
    

def cross_val_score(clf, X, y, n, n_jobs=1):
    skfolds = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)

    # create a list variable using multiprocessing PROCESS_MANAGER
    scores = PROCESS_MANAGER.list()

    for train_index, test_index in skfolds.split(X, y):
        clone_clf = clone(clf)
        X_train_folds = X[train_index]
        y_train_folds = y[train_index]
        X_test_folds = X[test_index]
        y_test_folds = y[test_index]

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_folds)
        n_correct = (y_pred == y_test_folds).sum()
        scores.append(n_correct / len(y_pred))

    return np.array(scores)

def display_cross_val_score(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())