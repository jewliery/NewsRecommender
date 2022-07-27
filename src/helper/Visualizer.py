from yellowbrick.text import FreqDistVisualizer
from yellowbrick.text import TSNEVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.cluster import KMeans


def showBarGraph(tweets, features):
    tweets = np.array(tweets)
    visualizer = FreqDistVisualizer(features=features)
    visualizer.fit(tweets)
    visualizer.show()


def showBar(tweets, features):
    tweets = np.array(tweets)
    visualizer = FreqDistVisualizer(features=features)
    visualizer.fit(tweets)
    visualizer.show()


def show2DVisualization(tweets):
    tweets = np.array(tweets)
    tsne = TSNEVisualizer()
    tsne.fit_transform(tweets)
    tsne.show()


def showAnother2DVisualization(tweets):
    corpus = []
    for tweet in tweets:
        t = tweet.text + "" + tweet.hashtags
        corpus.append(t)

    for c in corpus:
        print(c)
        print("------------------------------")
    tfidf = TfidfVectorizer()

    X = tfidf.fit_transform(corpus)

    clusters = KMeans(n_clusters=5)
    clusters.fit(X)

    tsne = TSNEVisualizer()
    tsne.fit(X, ["c{}".format(c) for c in clusters.labels_])
    tsne.show()
