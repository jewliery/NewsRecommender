from helper.Modeling import *
from helper.Modeling import evaluation
from helper.DataPreprocessor import UserData
from sklearn.metrics import pairwise_distances
from helper.Evaluation import Evaluation
import numpy as np

# Nimmt die Testdaten und zeigt welche Tweets empfohlen wurden
def getRecommendationList(pred, x_test, y_test, userData):
    recommendVectors = []
    recommended_y_value = []
    # Wenn 1 ist suche tweet und empfehle diesen
    for i in range(0, len(pred)):
        if pred[i] == 1:
            recommendVectors.append(x_test[i])
            recommended_y_value.append(y_test[i])

    # Finde die dazu passenden Tweets
    recommend = []
    for i in range(0, len(userData.x_train)):
        for r in recommendVectors:
            if userData.x_train[i] == r:
                recommend.append(userData.train[i])
                #userData.train[i].print(False)
    return recommend, recommendVectors, recommended_y_value

def getRecommendationVectors(pred, test):
    recommendVectors = []
    # Wenn 1 ist suche tweet und empfehle diesen
    for i in range(0, len(pred)):
        if pred[i] == 1:
            recommendVectors.append(test[i])
    return recommendVectors

def boundedGreedySelection(pred, x_test, y_test, userData, k, results):
    #evaluation.setResult("bgs", results)
    tweets, vec, y_all_values = getRecommendationList(pred, x_test, y_test, userData)
    y_values = []
    r = vec
    dist = getDistance(r)
    s = []

    # Add two furthest apart items to s and delete them from r
    maxi = 0.0
    tupel = [0,0]
    for i in range(0, len(dist)):
        for j in range(0, len(dist)):
            if dist[i,j] >= maxi:
                maxi = dist[i,j]
                tupel = [i,j]
    print(maxi, tupel)
    s.append(r[tupel[0]])
    s.append(r[tupel[1]])
    r.remove(r[tupel[0]])
    r.remove(r[tupel[1]])
    y_values.append(y_all_values[tupel[0]])
    y_values.append(y_all_values[tupel[1]])

    # Add item from r with maximum item-set distance from s till k is reached
    while len(s) < k:
        dist = getDistance(r, s)
        max_dist = 0
        index = 0
        for i in range(0, len(dist)):
            avg_dist = 0
            for j in range(0, len(dist[i])):
                avg_dist += dist[i,j]
            avg_dist = avg_dist/len(dist[i])
            if avg_dist >= max_dist:
                max_dist = avg_dist
            index = i
        s.append(r[index])
        y_values.append(y_all_values[index])
        r.remove(r[index])

    print("Y Values")
    print(y_values)
    pred_values = np.ones(k).tolist()
    results = evaluate(y_values, pred_values)
    evaluation.setResult("bgs", results)
    return s

def getDistance(x, y=[]):
    if len(y) != 0:
        distance = pairwise_distances(X=x, Y=y, metric='euclidean')
    else:
        distance = pairwise_distances(X=x,metric='euclidean')
    return distance


