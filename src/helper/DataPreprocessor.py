from helper.DataHelper import *
from helper.TwitterManager import TwitterManager
from helper.DataConverter import *
from deepdiff import DeepDiff
import numpy as np

# This File prepares the Data and finally splits it into Training and Test Data

api = TwitterManager()

# Collects all negative rated data from user,
def getNegativeDataFromUser(user_id):
    following = api.getAllFollowing(user_id)['data']
    all_tweets = np.array([])
    for u in following:
        tweets = api.getUserTimeline(int(u['id']))
        #tweets = api.getAllUsersTweets(int(u['id']))
        tweet_objects = convertTweetsToObjects(tweets)
        all_tweets = np.append(all_tweets, tweet_objects, axis=0)
    return all_tweets

# Collects all positively rated data from user
def getPositiveDataFromUser(user_id):
    tweets = api.getLikedTweets(user_id)
    tweetObjects = convertDictTweetsToObjects(tweets, addUser=True)
    return tweetObjects

# Collects all Data (Positive and Negative) from User
def getDataFromUser(user_name):
    user = api.getUser(user_name)
    user_id = user['data']['id']
    negative = getNegativeDataFromUser(user_id)
    positive = getPositiveDataFromUser(user_id)
    set_difference = set(negative) - set(positive)
    list_difference = list(set_difference)
    # list_difference = differenceOfTweetLists(negative, positive)
    return positive, list_difference

# Combines Positive and Negative Data and returns Vector of x_train and y_train
def getXYDataFromUser():
    return

#-----------------------------------Preprocessing-----------------------------------
# Checks if Tweets is two times, if yes remove from list
def differenceOfTweetLists(neg_tweets, pos_tweets):
    for neg_tweet in neg_tweets:
        for pos_tweet in pos_tweets:
            print(neg_tweet.id)
            if neg_tweet.id == pos_tweet.id:
                neg_tweets.remove(neg_tweet)
                print(neg_tweet)
    return neg_tweets

def convertDataToVector():
    return




