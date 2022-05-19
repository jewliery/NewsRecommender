from models.Rating import Rating
from models.UserObject import User
from models.TweetObject import Tweet
from helper.TwitterManager import TwitterManager

api = TwitterManager()

def isGermanTweet(tweet):
    try:
        if tweet['lang'] == 'de':
            return True
    except:
        if tweet.lang == 'de':
            return True
    return False

def convertTweetsToObjects(tweets):
    tweetObjects = []
    for tweet in tweets:
        if isGermanTweet(tweet):
            user = convertUserToObject(tweet.user)
            rating = Rating(tweet.favorited, tweet.retweeted, 0)
            hashtags = []
            for tag in tweet.entities['hashtags']:
                hashtags.append(tag['text'])
            tweetObject = Tweet(tweet.id, tweet.text, hashtags, user, False, tweet.lang, rating, False)
            tweetObjects.append(tweetObject)
    return tweetObjects

def convertDictTweetsToObjects(tweets, addUser=False):
    tweetObjects = []
    tweetsData = tweets['data']
    for tweet in tweetsData:
        if isGermanTweet(tweet):
            if addUser:
                u = api.getUser(user_id=tweet['author_id'])
                user = convertDictUserToObject(u)
            elif not addUser:
                user = User()
            hashtags = []
            try:
                for tag in tweet['entities']['hashtags']:
                    hashtags.append(tag['tag'])
            except:
                hashtags.append("")
            tweetObject = Tweet(id=tweet['id'], text=tweet['text'], hashtags=hashtags, user=user)
            tweetObjects.append(tweetObject)
    return tweetObjects


def convertUserToObject(user):
    userObject = User(user.id, user.screen_name, user.protected, user.followers_count,
                      user.verified)
    return userObject

def convertDictUserToObject(user):
    userData = user['data']
    try:
        userObject = User(id=userData['id'], name=userData['name'],
                      protected=userData['protected'],
                      followers=userData['public_metrics']['followers_count'],
                      verified=userData['verified'])
    except:
        userObject = User(id=userData['id'], name=userData['name'])
    return userObject