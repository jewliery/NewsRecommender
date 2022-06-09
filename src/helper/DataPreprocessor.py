from helper.DataHelper import *
from helper.TwitterManager import TwitterManager
from helper.DataConverter import *
from models.UserObject import UserObject
from deepdiff import DeepDiff
import numpy as np


class UserData:

    def __init__(self):
        self.api = TwitterManager()
        self.user = User()
        self.positiveTweets = []
        self.negativeTweets = []

    def setCurrentUser(self, user_name):
        self.user = self.api.getUser(user_name)

    # Collects all negative rated data from user,
    def getNegativeDataFromUser(self, user_id):
        following = self.api.getAllFollowing(user_id)['data']
        all_tweets = np.array([])
        for u in following:
            tweets = self.api.getUserTimeline(int(u['id']))
            #tweets = api.getAllUsersTweets(int(u['id']))
            tweet_objects = convertTweetsToObjects(tweets)
            all_tweets = np.append(all_tweets, tweet_objects, axis=0)
        return all_tweets

    # Collects all positively rated data from user
    def getPositiveDataFromUser(self, user_id):
        tweets = self.api.getLikedTweets(user_id)
        tweetObjects = convertDictTweetsToObjects(tweets, addUser=False)
        return tweetObjects

    # Collects all Data (Positive and Negative) from User
    def getDataFromUser(self, user_name):
        user = self.api.getUser(user_name)
        user_id = user['data']['id']
        negative = getNegativeDataFromUser(user_id)
        positive = getPositiveDataFromUser(user_id)
        set_difference = set(negative) - set(positive)
        list_difference = list(set_difference)
        # list_difference = differenceOfTweetLists(negative, positive)
        return positive, list_difference, user

    def getTrendsList(self):
        trends = self.api.getTrends()
        trend_list = []
        for value in trends:
            for trend in value['trends']:
                print(trend['name'])
                trend_list.append(trend['name'])
        return trend_list


    def getData(self):
        return


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





