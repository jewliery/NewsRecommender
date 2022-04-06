from models.Rating import Rating
from models.UserObject import User
from models.TweetObject import Tweet
from helper.TwitterManager import TwitterManager

user = User()
api = TwitterManager()

tweets = api.getFavoriteTweets()

for tweet in tweets:
    #user.addFavoriteTweet(tweet)
    print(tweet.text)
    user = User()
    rating = Rating(tweet.favorited, tweet.retweeted, 0)
    #Todo how to get matadata and possibly senstive attribute???
    tweetObject = Tweet(tweet.id, tweet.text, tweet.entities['hashtags'], user, False, tweet.lang, rating, False)
    #print(tweetObject)
