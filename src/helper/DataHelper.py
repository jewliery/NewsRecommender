from models.Rating import Rating
from models.UserObject import User
from models.TweetObject import Tweet
from helper.TwitterManager import TwitterManager
from helper.DataConverter import *
from helper.FeatureSelection import *
from sklearn import preprocessing
import numpy as np

api = TwitterManager()

#--------------------------Preprocessing-------------------------------
def scaleData(X_train):
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    return X_scaled

# Vorrübergehende Lösung
def createYdata(positive,neutral):
    pos = np.ones(len(positive),  dtype=int)
    neu = np.zeros(len(neutral),  dtype=int)
    y_train = np.concatenate([pos, neu], axis=0).tolist()
    return y_train

#----------------TESTS LATER PROBABLY IRRELEVANT--------------------------
#-------------------------API v1.1----------------------------------------
# Returns 20 Vectorized Tweets data which was rated positive by the user
def getMyLikedVectorData():
    tweets = api.getFavoriteTweets()
    tweetObjects = convertTweetsToObjects(tweets)
    vector, features = createVectors(tweetObjects)
    return vector

def getMyLikedRawData():
    tweets = api.getFavoriteTweets()
    tweetObjects = convertTweetsToObjects(tweets)
    return tweetObjects

# Returns Vectorized Tweets data which was not rated by the user
def getMyVectorTimeline():
    tweets = api.getTimeline()
    tweetObjects = convertTweetsToObjects(tweets)
    vector, features = createVectors(tweetObjects)
    return vector

# Returns Vectorized Tweets data which was rated positive by the user
def getMyLikedVectorDataTfIdf():
    tweets = api.getFavoriteTweets()
    tweetObjects = convertTweetsToObjects(tweets)
    vector = createVectorsTfIdf(tweetObjects)
    return vector


def getOverallData():
    positive_tweets = api.getFavoriteTweets()
    neutral_tweets = api.getTimeline()
    y_train = createYdata(positive_tweets, neutral_tweets)
    tweets = positive_tweets + neutral_tweets
    tweetObjects = convertTweetsToObjects(tweets)
    x_train, features = createVectors(tweetObjects)
    return x_train, y_train

#-------------------------API v2----------------------------------------
# Returns every Tweet I have liked as vector
def getAllMyLikedVectorData():
    tweets = api.getAllMyLikedTweets()
    tweetObjects = convertDictTweetsToObjects(tweets)
    vector, features = createVectors(tweetObjects)
    return vector

# Returns every Tweet I have liked as TweetObject
def getAllMyLikedRawData():
    tweets = api.getAllMyLikedTweets()
    tweetObjects = convertDictTweetsToObjects(tweets)
    return tweetObjects

# Returns Vectorized Tweets data which was rated positive by the user
def getAllLikedVectorDataTfIdf():
    tweets = api.getAllMyLikedTweets()
    tweetObjects = convertDictTweetsToObjects(tweets)
    vector = createVectorsTfIdf(tweetObjects)
    return vector












