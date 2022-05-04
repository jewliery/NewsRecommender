from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB

def train2DData(X, y):
    #y = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0])
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
    clf = MultinomialNB().fit(X_train, y_train)
    #lr.fit(X_train, y_train)
    #y_predict = lr.predict(X_test)
    y_predict = clf.predict(X_test)
    print("LogisticRegression Accuracy %.3f" % metrics.accuracy_score(y_test, y_predict))

def trainOCSVM(x):
    svm = OneClassSVM(nu=0.25, gamma=0.35)
    #svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)
    svm.fit(x)
    pred = svm.predict(x)
    print(pred)
    scores = svm.score_samples(x)
    print(scores)
    #thresh = np.quantile(scores, 0.03)
    #index = np.where(scores <= thresh)


def trainIsolationForest(x):
    return
