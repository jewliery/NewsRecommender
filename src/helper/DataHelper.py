from models.Rating import Rating
from models.UserObject import User
from models.TweetObject import Tweet
from helper.TwitterManager import TwitterManager
from helper.FeatureSelection import *
from sklearn import preprocessing
import numpy as np


# Returns Vectorized Tweets data which was rated positive by the user
def getTrainingsData():
    api = TwitterManager()
    tweets = api.getFavoriteTweets()
    tweetObjects = convertTweetsToObjects(tweets)
    vector = createVectors(tweetObjects)
    return vector

# Returns Vectorized Tweets data which was not rated by the user
def getTweetData():
    api = TwitterManager()
    tweets = api.getTimeline()
    tweetObjects = convertTweetsToObjects(tweets)
    vector = createVectors(tweetObjects)
    return vector


def getOverallData():
    api = TwitterManager()
    positive_tweets = api.getFavoriteTweets()
    neutral_tweets = api.getTimeline()
    y_train = combineData(positive_tweets, neutral_tweets)
    tweets = positive_tweets + neutral_tweets
    tweetObjects = convertTweetsToObjects(tweets)
    x_train = createVectors(tweetObjects)
    return x_train, y_train


def scaleData(X_train):
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    return X_scaled

def combineData(positive,neutral):
    pos = np.ones(len(positive),  dtype=int)
    neu = np.zeros(len(neutral),  dtype=int)
    y_train = np.concatenate([pos, neu], axis=0).tolist()
    return y_train









