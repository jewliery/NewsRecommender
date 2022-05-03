from models.Rating import Rating
from models.UserObject import User
from models.TweetObject import Tweet
from helper.TwitterManager import TwitterManager
from helper.FeatureSelection import *
from sklearn import preprocessing
import numpy as np
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# Returns Vectorized Tweets data which was rated positiv by the user
def getTrainingsData():
    api = TwitterManager()
    tweets = api.getFavoriteTweets()
    tweetObjects = convertTweetsToObjects(tweets)
    vector = createVectors(tweetObjects)
    return vector


def getTweetData():
    tweets = api.getTimeline()
    tweetObjects = convertTweetsToObjects(tweets)
    vector = createVectors(tweetObjects)
    return vector


def scaleData(X_train):
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    return X_scaled

def trainData(X):
    #y = np.ones(X.length)
    y = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0])
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
    clf = MultinomialNB().fit(X_train, y_train)
    #lr.fit(X_train, y_train)
    #y_predict = lr.predict(X_test)
    y_predict = clf.predict(X_test)
    print("LogisticRegression Accuracy %.3f" % metrics.accuracy_score(y_test, y_predict))
