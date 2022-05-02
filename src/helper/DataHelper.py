from models.Rating import Rating
from models.UserObject import User
from models.TweetObject import Tweet
from helper.TwitterManager import TwitterManager
from helper.FeatureSelection import *

#Returns Vectorized Tweets data which was rated positiv by the user
def getTrainingsData():
    tweets = api.getFavoriteTweets()
    tweetObjects = convertTweetsToObjects(tweets)
    vector = createVectors(tweetObjects)
    return vector


def getTweetData():
    tweets = api.getTimeline()
    tweetObjects = convertTweetsToObjects(tweets)
    vector = createVectors(tweetObjects)
    return vector
