from models.Rating import Rating
from models.UserObject import User
from models.TweetObject import Tweet
from helper.TwitterManager import TwitterManager
from helper.FeatureSelection import *
from helper.DataHelper import *

api = TwitterManager()

#tweets = api.getTweets("Ukraine")
#tweets = api.getTimeline()
tweets = api.getFavoriteTweets()
tweetObjects = convertTweetsToObjects(tweets)
numberOfTweets = 0
for tweet in tweetObjects:
    #user.addFavoriteTweet(tweet)
    tweet.print(False)
    numberOfTweets += 1

print("Number of Tweets: "+ str(numberOfTweets))
vector = getTrainingsData()
trainData(vector)
for vec in vector:
    print(vec)







