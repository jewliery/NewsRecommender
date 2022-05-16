from helper.DataHelper import *
from helper.TwitterManager import TwitterManager
from helper.DataConverter import *

# This File prepares the Data and finally splits it into Training and Test Data

# NOTE: Vektoren müssen zusammen erstellt werden, damit tfidf stimmt
# DAZU: Alle Daten sammeln und zusammen konvertieren - in der richtigen
#       Reihenfolge dann y_daten merken!

api = TwitterManager()

# Collects all negative rated data from user,
# Momentan alle Tweets von den Leuten denen man folgt
# TODO gleich diese mit geliketen Tweets ab
def getNegativeDataFromUser(user_name):
    user = api.getUser(user_name)
    userObject = convertDictUserToObject(user)
    following = api.getAllFollowing(userObject.id)
    all_tweets = []
    for u in following['data']:
        tweets = api.getAllUsersTweets(int(u['id']))
        all_tweets.append(tweets)
    tweetObjects = convertDictTweetsToObjects(all_tweets)
    return tweetObjects

# Checks if Tweets is two times, if yes remove from list
def checkData(tweets):
    return

# Collects all positively rated data from user
def getPositiveDataFromUser():
    return

def convertCollectedDataToVector():
    return

# Combines Positive and Negative Data
def getXYDataFromUser():
    return

