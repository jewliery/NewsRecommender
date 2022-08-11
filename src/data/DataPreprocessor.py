from data.DataConverter import *
import numpy as np


def createYdata(positive, neutral):
    pos = np.ones(len(positive), dtype=int)
    neu = np.zeros(len(neutral), dtype=int)
    y_train = np.concatenate([pos, neu], axis=0).tolist()
    return y_train


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

    def setTrainData(self, train, x_train, y_train):
        self.train = train
        self.y_train = y_train
        self.x_train = x_train

    def setCurrentUser(self, user_name):
        self.user = self.api.getUser(user_name)

    # Collects all negative rated data from user,
    def getNegativeDataFromUser(self, user_id, count):
        count = int(count / 4)
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
        tweetObjects, count = convertDictTweetsToObjects(tweets, addUser=True)  # HERE IF TOO MANY REQUESTS#
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

    def getData(self):
        all_tweets = np.array([])
        for u in self.following:
            tweets = self.api.getUserTimeline(int(u['id']))
            tweet_objects = convertTweetsToObjects(tweets)
            all_tweets = np.append(all_tweets, tweet_objects, axis=0)
        return all_tweets
