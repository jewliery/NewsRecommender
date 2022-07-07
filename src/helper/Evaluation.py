from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import numpy as np

class Metrics:
    def __init__(self, precision, recall, ild, novelty, exp):
        self.precision = precision
        self.recall = recall
        self.ild = ild
        self.exp = exp
        self.novelty = novelty

class Evaluation:
    def __init__(self):
        self.plain = []
        self.profile_partitioning = []
        self.bgs = []
        self.anomalies_exceptions = []

    def setResult(self, method="", result=[]):
        if method=="plain":
            self.plain = ["Plain"] + result
        if method=="profile_partitioning":
            self.profile_partitioning = ["Profile Partitioning"] + result
        if method=="bgs":
            self.bgs = ["Bounded-Greedy-Selection"] + result
        if method=="anomalies_exceptions":
            self.anomalies_exceptions = ["Anomalien und Ausnahmen"] + result

    def showResults(self):
        results = [self.plain, self.profile_partitioning, self.bgs, self.anomalies_exceptions]
        print(results)

        indices = np.arange(len(results))
        results = [[x[i] for x in results] for i in range(6)]

        clf_names, score, precision, recall, ild, unexp = results

        plt.figure(figsize=(12, 8))
        plt.title("Score")
        #plt.barh(indices, score, 0.15, label="Score", color="navy")
        plt.barh(indices, precision, 0.15, label="Precision", color="c")
        plt.barh(indices + 0.25, recall, 0.15, label="Recall", color="darkorange")
        plt.barh(indices + 0.5, ild , 0.15, label="ILD", color="red")
        plt.barh(indices + 0.75, unexp, 0.15, label="Unexp", color="navy")
        plt.yticks(())
        plt.legend(loc="best")
        plt.subplots_adjust(left=0.25)
        plt.subplots_adjust(top=0.95)
        plt.subplots_adjust(bottom=0.05)

        for i, c in zip(indices, clf_names):
            plt.text(-0.5, i, c) # vorher -0.3!

        plt.show()

    def getILD(self, x):
        distance = pairwise_distances(X=x, metric='euclidean')
        #print(distance)
        all_avg = 0
        for i in range(0,len(distance)-1):
            one_avg = 0
            for j in range(0, len(distance[i])-1):
                one_avg += distance[i][j]

            avg = one_avg/(len(distance[i])-1)
            all_avg += avg
        all_avg = all_avg/len(distance)
        all_avg -= 1
        return all_avg

    # Calculated difference between recommended items and known items
    def getUnexp(self, x, y):
        distance = pairwise_distances(X=x, Y=y, metric='euclidean')
        all_avg = 0
        for i in range(0, len(distance) - 1):
            one_avg = 0
            for j in range(0, len(distance[i]) - 1):
                one_avg += distance[i][j]

            avg = one_avg / (len(distance[i]))
            all_avg += avg
        all_avg = all_avg / len(distance)
        all_avg -= 1
        return all_avg



