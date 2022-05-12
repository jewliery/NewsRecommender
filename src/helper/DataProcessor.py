from models.Rating import Rating
from models.UserObject import User
from models.TweetObject import Tweet

def convertTweetsToObjects(tweets):
    tweetObjects = []
    for tweet in tweets:
        user = User(tweet.user.id, tweet.user.screen_name, tweet.user.protected, tweet.user.followers_count,
                    tweet.user.verified)
        rating = Rating(tweet.favorited, tweet.retweeted, 0)
        hashtags = []
        for tag in tweet.entities['hashtags']:
            hashtags.append(tag['text'])
        tweetObject = Tweet(tweet.id, tweet.text, hashtags, user, False, tweet.lang, rating, False)
        tweetObjects.append(tweetObject)
    return tweetObjects

def convertDictTweetsToObjects(tweets):
    tweetObjects = []
    for tweet in tweets['data']:
        hashtags = []
        try:
            for tag in tweet['entities']['hashtags']:
                hashtags.append(tag['tag'])
        except:
            hashtags.append("")
        tweetObject = Tweet(id=tweet['id'], text=tweet['text'], hashtags=hashtags)
        tweetObjects.append(tweetObject)
    return tweetObjects


def convertUserToObject(user):
    userObject = User(user.id, user.screen_name, user.protected, user.followers_count,
                      user.verified)
    return userObject