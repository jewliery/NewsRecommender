from helper.Modeling import *
from helper.Modeling import evaluation
from sklearn.metrics import pairwise_distances
import numpy as np


# Takes testdata and finds the corresponding tweets
def getRecList(pred, x_test, y_test):
    recommendVectors = []
    recommended_y_values = []
    not_recommended_y_values = []
    print("Original Prediction List")
    print(pred)
    # if 1, tweet is recommended
    for i in range(0, len(pred)):
        if pred[i] == 1:
            recommendVectors.append(x_test[i])
            recommended_y_values.append(y_test[i])
        if pred[i] == 0:
            not_recommended_y_values.append(y_test[i])

    return recommendVectors, recommended_y_values, not_recommended_y_values


def getRecommendationVectors(pred, test):
    recommendVectors = []
    # if 1, tweet is recommended
    for i in range(0, len(pred)):
        if pred[i] == 1:
            recommendVectors.append(test[i])
    return recommendVectors


def boundedGreedySelection(userData, k):
    clf, pred, x_test, y_test, results = createUserModel(userData, "bgs")
    vec, y_all_values, y_not_rec = getRecList(pred, x_test, y_test)
    y_values = []
    y_other = y_all_values
    print("Recommendation y Values")
    print(y_other)
    r = vec
    dist = getDistance(r)
    s = []

    # Add two furthest apart items to s and delete them from r
    maxi = 0.0
    tupel = [0, 0]
    for i in range(0, len(dist)):
        for j in range(0, len(dist)):
            if dist[i, j] >= maxi:
                maxi = dist[i, j]
                tupel = [i, j]
    print(maxi, tupel)
    s.append(r[tupel[0]])
    s.append(r[tupel[1]])
    r.remove(r[tupel[0]])
    r.remove(r[tupel[1]])
    y_values.append(y_all_values[tupel[0]])
    y_values.append(y_all_values[tupel[1]])
    y_other.remove(y_all_values[tupel[0]])
    y_other.remove(y_all_values[tupel[1]])

    # Add item from r with maximum item-set distance from s till k is reached
    while len(s) < k:
        dist = getDistance(r, s)
        max_dist = 0
        index = 0
        for i in range(0, len(dist)):
            avg_dist = 0
            for j in range(0, len(dist[i])):
                avg_dist += dist[i, j]
            avg_dist = avg_dist / len(dist[i])
            if avg_dist >= max_dist:
                max_dist = avg_dist
            index = i
        s.append(r[index])
        y_values.append(y_all_values[index])
        y_other.remove(y_all_values[index])
        r.remove(r[index])

    print(y_values)
    y = np.concatenate([y_values, y_other, y_not_rec], axis=0).tolist()
    pred = np.concatenate([np.ones(len(y_values), dtype=int), np.zeros(len(y_other) + len(y_not_rec), dtype=int)],
                          axis=0).tolist()
    print(y)
    print(pred)
    results = evaluate(y, pred)

    tweets = getTweets(s, userData)
    ild = evaluation.getILD(s)
    unexp = evaluation.getUnexp(s[0:10], userData.x_train[0:10])
    novelty = evaluation.getAvgNovelty(tweets[0:10])
    results.append(ild)
    results.append(unexp)
    results.append(novelty)
    evaluation.setResult("bgs", results)
    return s


def getDistance(x, y=[]):
    if len(y) != 0:
        distance = pairwise_distances(X=x, Y=y, metric='euclidean')
    else:
        distance = pairwise_distances(X=x, metric='euclidean')
    return distance
