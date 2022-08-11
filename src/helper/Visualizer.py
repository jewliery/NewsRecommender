from yellowbrick.text import FreqDistVisualizer
from yellowbrick.text import TSNEVisualizer
import numpy as np


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

