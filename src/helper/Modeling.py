from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import metrics
from helper.DataPreprocessor import *
from helper.DataHelper import *
from helper.FeatureSelection import *
from sklearn.utils.extmath import density
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from time import time
import matplotlib.pyplot as plt
import numpy as np

def getTrainingData(userData):
    positive, negative = userData.getDataFromUser()
    tweets = positive + negative
    y_train = createYdata(positive, negative)
    #x_train, features = createFullVectorsTfIdf(tweets)
    x_train, features = createVectorsTfIdf(tweets)
    userData.setTrainData(train=tweets, x_train=x_train, y_train=y_train)
    return x_train, y_train

def createUserModel(userData, algorithm):
    if userData.x_train or userData.y_train == []:
        x_train, y_train = getTrainingData(userData)
    else:
        x_train = userData.x_train
        y_train = userData.y_train
    if algorithm == "random-forest":
        clf, pred, test = randomForest(x_train, y_train)
    if algorithm == "naive-bayes":
        clf, pred, test = naiveBayes(x_train, y_train)
    if algorithm == "kNN":
        clf, pred, test = kNN(x_train, y_train)
    if algorithm == "SVM":
        clf, pred, test = SVM(x_train, y_train)
    if algorithm == "decision-tree":
        clf, pred, test = decisionTree(x_train, y_train)
    return clf, pred, test

def profilePartitioning(userData):
    if userData.x_train or userData.y_train == []:
        x_train, y_train = getTrainingData(userData)
    else:
        x_train = userData.x_train
        y_train = userData.y_train

    n = determineN(x_train)
    cluster = createCluster(x_train=x_train, n=n)
    print("Old Number of Clusters: " + str(n))

    # Create Lists for each Cluster and save y Value
    clustered_xvalue = []
    clustered_yvalue = []
    clustered_tweets = []
    for i in range(0,n):
        clustered_xvalue.append([])
        clustered_yvalue.append([])
        clustered_tweets.append([])

    for i in range(0, len(cluster)):
        for j in range(0, n):
            if cluster[i] == j:
                clustered_xvalue[j].append(x_train[i])
                clustered_yvalue[j].append(y_train[i])
                clustered_tweets[j].append(userData.train[i])
    print(clustered_yvalue)
    #printCluster(n,clustered_tweets)

    smallest_cluster = [] # List with indizes of Cluster size smaller than 6
    for i in range(0,n):
        if len(clustered_xvalue[i]) < 4:
            smallest_cluster.append(i)
            print("Index with small Cluster " + str(i))
    # Merge small clusters to one, do the same for y value
    bigger_cluster_x = []
    bigger_cluster_y = []
    bigger_cluster = []

    # Falls es ein Cluster gibt welches zu klein ist, verbinde es mit zweikleinsten Cluster
    if len(smallest_cluster) > 0:
        for s in smallest_cluster:
            bigger_cluster_x += clustered_xvalue[s]
            bigger_cluster_y += clustered_yvalue[s]
            bigger_cluster += clustered_tweets[s]
        for s in smallest_cluster:
            clustered_xvalue.remove(clustered_xvalue[s])
            clustered_yvalue.remove(clustered_yvalue[s])
            clustered_tweets.remove(clustered_tweets[s])

        index = 1000
        for i in range(0,n-1):
            if len(clustered_xvalue[i])<index:
                index = i

        clustered_xvalue[index] += bigger_cluster_x
        clustered_yvalue[index] += bigger_cluster_y
        clustered_tweets[index] += bigger_cluster
        n = len(clustered_xvalue)
        print("New Number of Clusters: " + str(n))

    #printCluster(n, clustered_tweets)

    results = []
    clf = DecisionTreeClassifier(max_depth=10)
    #clf = RandomForestClassifier()
    for i in range(0,n):
        result = benchmark(clf, clustered_xvalue[i], clustered_yvalue[i])
        results.append(result)
        print(result)

    avg_results = averageResults(results)
    results.append(avg_results)
    showResults(results)

def anomaliesExceptions():
    if userData.x_train or userData.y_train == []:
        x_train, y_train = getTrainingData(userData)
    else:
        x_train = userData.x_train
        y_train = userData.y_train
    clf, pred, test = naiveBayesProbs(x_train, y_train)


def printCluster(n, clustered_tweets):
    for i in range(0, n):
        print("-------------------" + str(i) + "tes Cluster ----------------------")
        for tweet in clustered_tweets[i]:
            tweet.print(False)

def determineN(x_train):
    distortions = []
    for i in range(1,6):
        km = KMeans(n_clusters=i,
                    init='random',
                    n_init=10,
                    max_iter=300,
                    random_state=0)
        km.fit(x_train)
        distortions.append(km.inertia_)
    current_n = 0
    for i in range(0,3):
        if distortions[i]-distortions[i+1] > distortions[i+1]-distortions[i+2]:
            current_n = i+1
        else:
            current_n = 2
    plt.plot(range(1,6), distortions, marker='o')
    plt.xlabel('Anzahl der Cluster')
    plt.ylabel('Verzerrung')
    plt.show()
    return current_n

def createCluster(x_train, n=3):
    km = KMeans(n_clusters=n,
                init='random',
                n_init=10,
                max_iter=300,
                tol=1e-02,
                random_state=0)
    y_km = km.fit_predict(x_train)
    print("Verzerrung: %.2f" % km.inertia_)
    return y_km

# TODO Schaue im Buch nach Parametern
def SVM(x,y):
    clf = SVC(gamma=2, C=1)
    clf, pred, test = bench(clf, x, y)
    return clf, pred, test

def naiveBayesProbs(x,y):
    clf = MultinomialNB(alpha=0.01)
    clf, pred, pred_proba, test = bench_proba(clf, x, y)
    #sort_probs_elements(pred_proba, test)
    sort_diff_elements(pred_proba, test)
    return clf, pred, test

def naiveBayes(x,y):
    clf = MultinomialNB(alpha=0.01)
    clf, pred, test = bench(clf, x, y)
    return clf, pred, test

def sort_probs_elements(pred, X):
    x_prob = []
    for i in range(0, len(X)):
        key_val = X[i]
        key_val.append(pred[i][0])
        key_val.append(pred[i][1])
        x_prob.append(key_val)

    sorted_list = sorted(x_prob, key=lambda x: x[-1], reverse=True)
    for el in sorted_list:
        print(el)

def sort_diff_elements(pred, X):
    x_prob = []
    for i in range(0, len(X)):
        key_val = X[i]
        key_val.append(abs(pred[i][1]-pred[i][0]))
        x_prob.append(key_val)

    sorted_list = sorted(x_prob, key=lambda x: x[-1])
    for el in sorted_list:
        print(el)


# Larger n numbers of neighbors means less noise but makes the classification boundaries less distinct
def kNN(x,y):
    clf = KNeighborsClassifier(n_neighbors=10)
    clf, pred, test = bench(clf, x, y)
    return clf, pred, test

def randomForest(x,y):
    clf = RandomForestClassifier()
    clf, pred, test = bench(clf, x, y)
    return clf, pred, test

# The maximum depth of the tree
def decisionTree(x,y):
    clf = DecisionTreeClassifier(max_depth=5)
    clf, pred, test = bench(clf, x, y)
    return clf, pred, test

def bench(clf, x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    print("_" * 80)
    print("Training: ")
    print(clf)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    print("Prediction: " % pred)
    print(pred)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    print("classification report:")
    print(metrics.classification_report(y_test, pred))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    return clf, pred, X_test

def bench_proba(clf, x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    print("_" * 80)
    print("Training: ")
    print(clf)
    clf.fit(X_train, y_train)

    pred_proba = clf.predict_proba(X_test)
    print("Prediction: " % pred_proba)
    print(pred_proba)

    pred = clf.predict(X_test)
    print("Prediction: " % pred)
    print(pred)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    print("classification report:")
    print(metrics.classification_report(y_test, pred))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    return clf, pred, pred_proba, X_test

def benchmark(clf, x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    print("_" * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    print("classification report:")
    print(metrics.classification_report(y_test, pred))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split("(")[0]
    return clf_descr, score, train_time, test_time

def averageResults(results):
    n = 0
    score = 0
    train_time = 0
    test_time = 0
    for r in results:
        score += r[1]
        train_time += r[2]
        test_time += r[3]
        n += 1
    avg_results = ["Average Results", score/n, train_time/n, test_time/n]
    return avg_results

def testModels(user_name):
    positive, negative = getDataFromUser(user_name)
    tweets = positive + negative
    y_train = createYdata(positive, negative)
    x_train, features = createVectorsTfIdf(tweets)

    results = []
    for clf, name in (
            (KNeighborsClassifier(n_neighbors=10), "kNN"),
            (RandomForestClassifier(), "Random forest"),
            (MultinomialNB(alpha=0.01), "Naiver-Bayes, Multinominal"),
            (DecisionTreeClassifier(max_depth=5), "Decision Tree Classifier"),
            (SVC(gamma=2, C=1), "SVM")

    ):
        print("=" * 80)
        print(name)
        results.append(benchmark(clf, x_train, y_train))

    showResults(results)


def showResults(results):
    indices = np.arange(len(results))
    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, 0.2, label="score", color="navy")
    plt.barh(indices + 0.3, training_time, 0.2, label="training time", color="c")
    plt.barh(indices + 0.6, test_time, 0.2, label="test time", color="darkorange")
    plt.yticks(())
    plt.legend(loc="best")
    plt.subplots_adjust(left=0.25)
    plt.subplots_adjust(top=0.95)
    plt.subplots_adjust(bottom=0.05)

    for i, c in zip(indices, clf_names):
        plt.text(-0.3, i, c)

    plt.show()