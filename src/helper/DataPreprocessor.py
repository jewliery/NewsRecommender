from helper.DataHelper import *
from helper.TwitterManager import TwitterManager
from helper.DataConverter import *
from models.UserObject import User
from deepdiff import DeepDiff
import numpy as np


class UserData:
    def __init__(self, user_name):
        self.api = TwitterManager()
        self.user = convertDictUserToObject(self.api.getUser(user_name))
        self.following = []
        self.positiveTweets = []
        self.negativeTweets = []

        self.train = []
        self.x_train = []
        self.y_train = []

    def setTrainData(self, train=[], x_train=[], y_train=[]):
        self.train = train
        self.y_train = y_train
        self.x_train = x_train

    def setCurrentUser(self, user_name):
        self.user = self.api.getUser(user_name)

    # Collects all negative rated data from user,
    def getNegativeDataFromUser(self, user_id, count):
        count = int(count/4) + 3
        self.following = self.api.getAllFollowing(user_id=user_id, max_results=count)['data']
        all_tweets = np.array([])
        for u in self.following:
            tweets = self.api.getUserTimeline(int(u['id']))
            tweet_objects = convertTweetsToObjects(tweets)
            all_tweets = np.append(all_tweets, tweet_objects, axis=0)
        return all_tweets

    # Collects all positively rated data from user
    def getPositiveDataFromUser(self, user_id):
        tweets = self.api.getLikedTweets(user_id)
        tweetObjects, count = convertDictTweetsToObjects(tweets, addUser=True) # HERE IF TOO MANY REQUESTS
        return tweetObjects, count

    # Collects all Data (Positive and Negative) from User
    def getDataFromUser(self):
        positive, count = self.getPositiveDataFromUser(self.user.id)
        negative = self.getNegativeDataFromUser(self.user.id, count)
        set_difference = set(negative) - set(positive)
        self.positiveTweets = positive
        self.negativeTweets = list(set_difference)
        return self.positiveTweets, self.negativeTweets

    def getTrendsList(self):
        trends = self.api.getTrends()
        trend_list = []
        for value in trends:
            for trend in value['trends']:
                print(trend['name'])
                trend_list.append(trend['name'])
        return trend_list

# Soll Menge liefern, welche der User noch nicht kennt
    def getData(self):
        all_tweets = np.array([])
        for u in self.following:
            tweets = self.api.getUserTimeline(int(u['id']))
            #tweets = api.getAllUsersTweets(int(u['id']))
            tweet_objects = convertTweetsToObjects(tweets)
            all_tweets = np.append(all_tweets, tweet_objects, axis=0)
        return all_tweets


    #-----------------------------------Preprocessing-----------------------------------
    # Checks if Tweets is two times, if yes remove from list
    def differenceOfTweetLists(self, neg_tweets, pos_tweets):
        for neg_tweet in neg_tweets:
            for pos_tweet in pos_tweets:
                print(neg_tweet.id)
                if neg_tweet.id == pos_tweet.id:
                    neg_tweets.remove(neg_tweet)
                    print(neg_tweet)
        return neg_tweets

    def scaleData(X_train):
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_scaled = scaler.transform(X_train)
        return X_scaled

    # Vorrübergehende Lösung
    def createYdata(self, positive, neutral):
        pos = np.ones(len(positive), dtype=int)
        neu = np.zeros(len(neutral), dtype=int)
        y_train = np.concatenate([pos, neu], axis=0).tolist()
        return y_train





