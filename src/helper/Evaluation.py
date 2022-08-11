import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import numpy as np


class Evaluation:
    def __init__(self):
        self.plain = []
        self.profile_partitioning = []
        self.bgs = []
        self.anomalies_exceptions = []

    def setResult(self, method="", result=[]):
        if method == "plain":
            self.plain = ["Plain"] + result
        if method == "profile_partitioning":
            self.profile_partitioning = ["Profile Partitioning"] + result
        if method == "bgs":
            self.bgs = ["Bounded-Greedy-Selection"] + result
        if method == "anomalies_exceptions":
            self.anomalies_exceptions = ["Anomalien und Ausnahmen"] + result

    def showResults(self):
        results = [self.plain, self.profile_partitioning, self.bgs, self.anomalies_exceptions]

        for i in results:
            self.printResults(i, i[0])

        indices = np.arange(len(results))
        results = [[x[i] for x in results] for i in range(7)]

        clf_names, score, precision, recall, ild, unexp, novelty = results

        plt.figure(figsize=(12, 8))
        plt.title("Evaluierung")
        plt.barh(indices, precision, 0.15, label="Precision", color="c")
        plt.barh(indices + 0.2, recall, 0.15, label="Recall", color="darkorange")
        plt.barh(indices + 0.4, ild, 0.15, label="ILD", color="red")
        plt.barh(indices + 0.6, unexp, 0.15, label="Unexp", color="navy")
        plt.barh(indices + 0.8, novelty, 0.15, label="Novelty", color="blue")
        plt.yticks(())
        plt.legend(loc="best")
        plt.subplots_adjust(left=0.25)
        plt.subplots_adjust(top=0.95)
        plt.subplots_adjust(bottom=0.05)

        for i, c in zip(indices, clf_names):
            plt.text(-1.0, i, c)

        plt.show()

    @staticmethod
    def getILD(x):
        distance = pairwise_distances(X=x, metric='euclidean')
        all_avg = 0
        for i in range(0, len(distance) - 1):
            one_avg = 0
            for j in range(0, len(distance[i]) - 1):
                one_avg += distance[i][j]

            avg = one_avg / (len(distance[i]) - 1)
            all_avg += avg
        all_avg = all_avg / len(distance)
        all_avg -= 1
        return all_avg

    # Calculated difference between recommended items and known items
    @staticmethod
    def getUnexp(x, y):
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

    def getAvgNovelty(self, tweets):
        iuf = 0
        for t in tweets:
            iuf += self.getIUF(t, tweets)
        avg_novelty = -(1 / len(tweets)) * iuf
        return avg_novelty

    @staticmethod
    def getIUF(tweet, tweets):
        user_count = 0
        for t in tweets:
            user_count += t.popularity

        popularity = tweet.popularity
        if user_count > 0:
            x = popularity / user_count
            iuf = np.log(x)
        else:
            iuf = 0
        return iuf

    def showResult(self, method):
        if method == "bgs":
            results = self.bgs
        elif method == "upp":
            results = self.profile_partitioning
        elif method == "aua":
            results = self.anomalies_exceptions
        elif method == "plain":
            results = self.plain
        else:
            results = self.plain

        self.printResults(results, method)

        clf_names, score, precision, recall, ild, unexp, novelty = results

        plt.figure(figsize=(12, 8))
        plt.title("Score")
        plt.barh(0, score, 0.2, label="Score", color="navy")
        plt.barh(1, precision, 0.2, label="Precision", color="c")
        plt.barh(2, recall, 0.2, label="Recall", color="darkorange")
        plt.barh(3, ild, 0.2, label="ILD", color="red")
        plt.barh(4, unexp, 0.2, label="Unexp", color="navy")
        plt.barh(5, novelty, 0.2, label="Novelty", color="blue")
        plt.yticks(())
        plt.legend(loc="best")
        plt.subplots_adjust(left=0.25)
        plt.subplots_adjust(top=0.95)
        plt.subplots_adjust(bottom=0.05)

        plt.show()

    @staticmethod
    def printResults(results, method):
        print("------Results from " + method + ":------")
        print("Score:   %0.3f" % results[1])
        print("Precision:   %0.3f" % results[2])
        print("Recall:   %0.3f" % results[3])
        print("ILD:   %0.3f" % results[4])
        print("Unexp:   %0.3f" % results[5])
        print("Novelty:   %0.3f" % results[6])



