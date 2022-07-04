from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

class Metrics:
    def __init__(self, precision, recall, ild, novelty, exp):
        self.precision = precision
        self.recall = recall
        self.ild = ild
        self.novelty = novelty
        self.exp = exp



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
        results = [[x[i] for x in results] for i in range(4)]

        clf_names, score, precision, recall = results

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
            plt.text(-0.3, i, c)

        plt.show()



