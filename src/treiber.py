from models.Rating import Rating
from models.UserObject import User
from models.TweetObject import Tweet
from helper.TwitterManager import TwitterManager
from helper.FeatureSelection import *

user = User()
api = TwitterManager()

#tweets = api.getFavoriteTweets()
tweets = api.getTimeline()
#tweets = api.getTweets("Ukraine")
tweetObjects = convertTweetsToObjects(tweets)

for tweet in tweetObjects:
    #user.addFavoriteTweet(tweet)
    tweet.print(False)

vector = createVectors(tweetObjects)
print(vector)





