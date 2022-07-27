from sklearn.model_selection import train_test_split
from sklearn import metrics
from data.DataPreprocessor import *
from data.FeatureSelection import *
from helper.Evaluation import Evaluation
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import pairwise_distances
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

evaluation = Evaluation()


def showEvaluation():
    evaluation.showResults()


def showResult():
    evaluation.showResult()


def getTrainingData(userData):
    positive, negative = userData.getDataFromUser()
    tweets = positive + negative
    y_train = createYdata(positive, negative)
    x_train, features = createFullVectorsTfIdf(tweets)
    userData.setTrainData(train=tweets, x_train=x_train, y_train=y_train)
    return x_train, y_train


def createUserModel(userData, algorithm):
    if userData.x_train or userData.y_train == []:
        x_train, y_train = getTrainingData(userData)
    else:
        x_train = userData.x_train
        y_train = userData.y_train
    if algorithm == "random-forest":
        clf, pred, x_test, y_test, results = randomForest(x_train, y_train)
    elif algorithm == "naive-bayes":
        clf, pred, x_test, y_test, results = naiveBayes(x_train, y_train)
    elif algorithm == "kNN":
        clf, pred, x_test, y_test, results = kNN(x_train, y_train)
    elif algorithm == "SVM":
        clf, pred, x_test, y_test, results = SVM(x_train, y_train)
    elif algorithm == "decision-tree":
        clf, pred, x_test, y_test, results = decisionTree(x_train, y_train)
    elif algorithm == "bgs":
        clf, pred, x_test, y_test, results = naiveBayes(x_train, y_train)
    else:
        clf, pred, x_test, y_test, results = naiveBayes(x_train, y_train)

    if algorithm != "bgs":
        rec_list, rec_tweets = getRecommendationList(pred, x_test, userData)
        ild = evaluation.getILD(rec_list[0:8])
        unexp = evaluation.getUnexp(rec_list[0:10], userData.x_train[0:10])
        novelty = evaluation.getAvgNovelty(rec_tweets[0:10])
        results.append(ild)
        results.append(unexp)
        results.append(novelty)
        evaluation.setResult("plain", results)

    return clf, pred, x_test, y_test, results


# --------------------------Profile Partitioning---------------------------#
def profile_partitioning(userData, k):
    if userData.x_train or userData.y_train == []:
        getTrainingData(userData)
    x_positiveTweets = userData.positiveTweets
    x_negativeTweets = userData.negativeTweets

    X, features = createFullVectorsTfIdf(x_positiveTweets + x_negativeTweets)
    x_positive = X[0:len(x_positiveTweets)]
    x_negative = X[-len(x_negativeTweets):]

    y_positive = np.ones(len(x_positive), dtype=int)
    x_train_pos, x_test_pos, y_train_pos, y_test_pos = train_test_split(x_positive, y_positive)

    n = determineN(x_train_pos)
    cluster = createCluster(x_train=x_train_pos, n=n)

    # Erstelle Cluster Liste, wobei x_cluster[i] das i-te Cluster enthält
    x_cluster = []
    for i in range(0, n):
        x_cluster.append([])

    for i in range(0, len(cluster)):
        for j in range(0, n):
            if cluster[i] == j:
                x_cluster[j].append(x_train_pos[i])

    # Erstelle Menge aus welchen Tweets empfohlen werden sollen
    x_neg = x_negative[0:len(x_test_pos)]
    y_neg = np.zeros(len(y_test_pos), dtype=int)
    x_test = np.concatenate([x_test_pos, x_neg], axis=0).tolist()
    y_test = np.concatenate([y_test_pos, y_neg], axis=0).tolist()

    m = k / n  # Anzahl von Tweets welche aus jedem Cluster hinzugefügt werden sollen
    m = int(m)

    x_nearest_tweets = []
    y_nearest_tweets = []
    y_other_tweets = []
    # Für jedes Cluster die Elemente welche dem Cluster am nächsten sind
    for i in range(0, n):
        x = x_cluster[i]
        distance = pairwise_distances(X=x, Y=x_test, metric='euclidean')
        avg_dist = getAvgDistances(distance)
        avg_dist, x_test, y_test = iter(avg_dist), iter(x_test), iter(y_test)
        sorted_x_test = [i for _, i in sorted(zip(avg_dist, x_test))]
        sorted_y_test = [i for _, i in sorted(zip(avg_dist, y_test))]
        x_nearest_tweets += sorted_x_test[0:m]  # Die Tweets die empfohlen werden sollen
        y_nearest_tweets += sorted_y_test[0:m]  # Die "wahren" Werte, ob Tweet relevant ist oder nicht
        y_other_tweets += sorted_y_test[m:]

    y_sorted = np.concatenate([y_nearest_tweets, y_other_tweets], axis=0).tolist()
    y_pred = np.concatenate([np.ones(len(y_nearest_tweets), dtype=int), np.zeros(len(y_other_tweets), dtype=int)],
                            axis=0).tolist()
    print("Recommended Tweets and if they are relevant or not")
    print(y_nearest_tweets)

    tweets = getTweets(x_nearest_tweets[0:10], userData)
    results = evaluate(y_sorted, y_pred)

    ild = evaluation.getILD(x_nearest_tweets)
    unexp = evaluation.getUnexp(x_nearest_tweets[0:10], userData.x_train[0:10])
    novelty = evaluation.getAvgNovelty(tweets)
    results.append(ild)
    results.append(unexp)
    results.append(novelty)
    evaluation.setResult("profile_partitioning", results)


def printCluster(n, clustered_tweets):
    for i in range(0, n):
        print("-------------------" + str(i) + "tes Cluster ----------------------")
        for tweet in clustered_tweets[i]:
            tweet.print(False)


def determineN(x_train):
    distortions = []
    for i in range(1, 6):
        km = KMeans(n_clusters=i,
                    init='random',
                    n_init=10,
                    max_iter=300,
                    random_state=0)
        km.fit(x_train)
        distortions.append(km.inertia_)
    current_n = 0
    for i in range(0, 3):
        if distortions[i] - distortions[i + 1] > distortions[i + 1] - distortions[i + 2]:
            current_n = i + 1
        else:
            current_n = 2
    # plt.plot(range(1,6), distortions, marker='o')
    # plt.xlabel('Anzahl der Cluster')
    # plt.ylabel('Verzerrung')
    # plt.show()
    return current_n


def getAvgDistances(distance):
    avg_distances = []
    for i in range(0, len(distance)):
        avg_dist = 0
        for j in range(0, len(distance[i])):
            avg_dist += distance[i][j]
        avg_distances.append(avg_dist / len(distance[i]))
    return avg_distances


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


# ---------------------------Anomalies Exceptions-------------------------#

def anomaliesExceptions(userData, k):
    if userData.x_train or userData.y_train == []:
        x_train, y_train = getTrainingData(userData)
    else:
        x_train = userData.x_train
        y_train = userData.y_train
    clf, pred, pred_proba, test_x, test_y = naiveBayesProbs(x_train, y_train)
    sorted_list, y_real = sort_diff_elements(pred_proba, test_x, test_y)
    y_pred = np.zeros(len(sorted_list)).tolist()
    for i in range(0, k):
        sorted_list[i].pop()  # Letztes Element entfernen - pred value
        sorted_list[i].pop(0)  # Erstes Element entfernen - y value
        y_pred[i] = 1  # Fill with ones, because the items are recommended
    recommendation_list = sorted_list[0:k]
    recommendation_tweets = getTweets(recommendation_list, userData)
    results = evaluate(y_real, y_pred)
    ild = evaluation.getILD(recommendation_list)
    unexp = evaluation.getUnexp(recommendation_list[0:10], userData.x_train[0:10])
    novelty = evaluation.getAvgNovelty(recommendation_tweets[0:10])
    results.append(ild)
    results.append(unexp)
    results.append(novelty)
    evaluation.setResult("anomalies_exceptions", results)
    return recommendation_list


def evaluate(y_test, pred):
    results = []
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    results.append(score)

    precision = metrics.precision_score(y_test, pred)
    print("precision:   %0.3f" % precision)
    results.append(precision)

    recall = metrics.recall_score(y_test, pred)
    print("recall:   %0.3f" % recall)
    results.append(recall)

    return results


def naiveBayesProbs(x, y):
    clf = MultinomialNB(alpha=0.01)
    clf, pred, pred_proba, test_x, test_y = bench_proba(clf, x, y)
    return clf, pred, pred_proba, test_x, test_y


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


def sort_diff_elements(pred, X, y):
    x_prob = []
    y_test = []
    for i in range(0, len(X)):
        key_val = [y[i]] + X[i]
        key_val.append(abs(pred[i][1] - pred[i][0]))
        x_prob.append(key_val)

    sorted_list = sorted(x_prob, key=lambda x: x[-1])

    for i in range(0, len(sorted_list)):
        y_test.append(sorted_list[i][0])  # Create List with real y values

    return sorted_list, y_test


# --------------------------------Classifier--------------------------------#
# Larger n numbers of neighbors means less noise but makes the classification boundaries less distinct
def kNN(x, y):
    clf = KNeighborsClassifier(n_neighbors=10)
    clf, pred, x_test, y_test, results = bench(clf, x, y)
    return clf, pred, x_test, y_test, results


def naiveBayes(x, y):
    clf = MultinomialNB()
    clf, pred, x_test, y_test, results = bench(clf, x, y)
    return clf, pred, x_test, y_test, results


def SVM(x, y):
    clf = SVC(gamma=2, C=1)
    clf, pred, x_test, y_test, results = bench(clf, x, y)
    return clf, pred, x_test, y_test, results


def randomForest(x, y):
    clf = RandomForestClassifier()
    clf, pred, x_test, y_test, results = bench(clf, x, y)
    return clf, pred, x_test, y_test, results


# The maximum depth of the tree
def decisionTree(x, y):
    clf = DecisionTreeClassifier(max_depth=5)
    clf, pred, x_test, y_test, results = bench(clf, x, y)
    return clf, pred, x_test, y_test, results


# --------------------------------Benchmarks--------------------------------#

def bench(clf, x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    print("_" * 80)
    print("Data:")
    print(y_train)
    print(y_test)
    print("Training: ")
    print(clf)
    clf.fit(X_train, y_train)
    results = []

    pred = clf.predict(X_test)
    print("Prediction: " % pred)

    score = metrics.accuracy_score(y_test, pred)
    print("Accuracy:   %0.3f" % score)
    results.append(score)

    precision = metrics.precision_score(y_test, pred)
    print("Precision:   %0.3f" % precision)
    results.append(precision)

    recall = metrics.recall_score(y_test, pred)
    print("Recall:   %0.3f" % recall)
    results.append(recall)

    print("Classification report:")
    print(metrics.classification_report(y_test, pred))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    return clf, pred, X_test, y_test, results


def bench_proba(clf, x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    print("_" * 80)
    print("Data:")
    print(y_train)
    print(y_test)
    print("Training: ")
    print(clf)
    clf.fit(X_train, y_train)

    pred_proba = clf.predict_proba(X_test)
    print("Prediction: " % pred_proba)
    print(pred_proba)

    pred = clf.predict(X_test)
    print("Prediction: " % pred)

    precision = metrics.precision_score(y_test, pred)
    print("Precision:   %0.3f" % precision)

    recall = metrics.recall_score(y_test, pred)
    print("Recall:   %0.3f" % recall)

    score = metrics.accuracy_score(y_test, pred)
    print("Accuracy:   %0.3f" % score)

    print("classification report:")
    print(metrics.classification_report(y_test, pred))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    return clf, pred, pred_proba, X_test, y_test


def benchmark(clf, x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    print("_" * 80)
    print("Data:")
    print(y_train)
    print(y_test)
    print("Training: ")
    print(clf)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    print("Prediction: " % pred)

    score = metrics.accuracy_score(y_test, pred)
    print("Accuracy:   %0.3f" % score)

    print("classification report:")
    print(metrics.classification_report(y_test, pred))

    precision = metrics.precision_score(y_test, pred)
    print("Precision:   %0.3f" % precision)

    recall = metrics.recall_score(y_test, pred)
    print("Recall:   %0.3f" % recall)

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split("(")[0]
    return pred, X_test, score, precision, recall, clf_descr


def averageResults(results):
    n = 0
    score = 0
    precision = 0
    recall = 0
    for r in results:
        score += r[0]
        precision += r[1]
        recall += r[2]
        n += 1
    avg_results = [score / n, precision / n, recall / n]
    return avg_results


def testModels(userData):
    if userData.x_train or userData.y_train == []:
        x_train, y_train = getTrainingData(userData)
    else:
        x_train = userData.x_train
        y_train = userData.y_train

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

    showClassifierResults(results)


def showClassifierResults(results):
    indices = np.arange(len(results))
    results = [[x[i] for x in results] for i in range(6)]

    # clf_names, score, precision, recall = results
    pred, X_test, score, precision, recall, clf_names = results

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, 0.2, label="Score", color="navy")
    plt.barh(indices + 0.3, precision, 0.2, label="Precision", color="c")
    plt.barh(indices + 0.6, recall, 0.2, label="Recall", color="darkorange")
    plt.yticks(())
    plt.legend(loc="best")
    plt.subplots_adjust(left=0.25)
    plt.subplots_adjust(top=0.95)
    plt.subplots_adjust(bottom=0.05)

    for i, c in zip(indices, clf_names):
        plt.text(-0.25, i, c)

    plt.show()


def getRecommendationList(pred, x_test, userData):
    recommendVectors = []
    for i in range(0, len(pred)):
        if pred[i] == 1:
            recommendVectors.append(x_test[i])
    recommend = []
    for i in range(0, len(userData.x_train)):
        for r in recommendVectors:
            if userData.x_train[i] == r:
                recommend.append(userData.train[i])
    return recommendVectors, recommend


def getTweets(vectors, userData):
    recommend = []
    for i in range(0, len(userData.x_train)):
        for r in vectors:
            if userData.x_train[i] == r:
                recommend.append(userData.train[i])
    return recommend
