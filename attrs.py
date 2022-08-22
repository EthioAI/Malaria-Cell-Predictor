<<<<<<< HEAD
import enum
=======
>>>>>>> 53d1c7ad4d536ba8a8b9005fda0fbb2e2be07d0f
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from skimage import color
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
<<<<<<< HEAD
from sklearn.base import BaseEstimator, TransformerMixin, clone
=======
from sklearn.base import clone
>>>>>>> 53d1c7ad4d536ba8a8b9005fda0fbb2e2be07d0f
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
<<<<<<< HEAD
            img = cv2.resize(plt.imread(path), (50,50))
            # img = img.reshape(-1, 3)
            X.append(img)
            y.append(1.)
=======
            img = cv2.resize(plt.imread(path), (25,25))
            img = color.rgb2gray(img).ravel()
            X.append(img)
            y.append(1)
>>>>>>> 53d1c7ad4d536ba8a8b9005fda0fbb2e2be07d0f

    for i in os.listdir("data/cell_images/Uninfected/"):
        if ".png" in i:
            path = "data/cell_images/Uninfected/"+i
<<<<<<< HEAD
            img = cv2.resize(plt.imread(path), (50,50))
            # img = img.reshape(-1, 3)
            X.append(img)
            y.append(0.)
=======
            img = cv2.resize(plt.imread(path), (25,25))
            img = color.rgb2gray(img).ravel()
            X.append(img)
            y.append(0)
>>>>>>> 53d1c7ad4d536ba8a8b9005fda0fbb2e2be07d0f

    X = np.array(X) 
    y = np.array(y)

    return X, y
<<<<<<< HEAD


class ArrayRavel(BaseEstimator, TransformerMixin):
    '''
    It changes the arrays in the dataframe to be 1D arrays.
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X:np.ndarray):
        X_raveled = np.array([x.ravel() for x in X])
        return X_raveled
=======
>>>>>>> 53d1c7ad4d536ba8a8b9005fda0fbb2e2be07d0f
    

def cross_val_score(clf, X, y, n):
    skfolds = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
    scores = []

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

<<<<<<< HEAD
def display_cross_val_score(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())
=======
def display_cross_val_score(scores, write_log=False, write_path="cross_val_score.log"):
    mean = scores.mean()
    std = scores.std()

    output = f"Scores: {scores}\n"
    output += f"Mean: {mean}\n"
    output += f"Standard deviation: {std}"
    print(output)

    if write_log:
        with open(write_path, "w+") as f:
            f.write(output)
            f.close()





>>>>>>> 53d1c7ad4d536ba8a8b9005fda0fbb2e2be07d0f
