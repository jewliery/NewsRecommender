from models.UserObject import User
from models.TweetObject import Tweet
from data.TwitterManager import TwitterManager

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
            hashtags = []
            for tag in tweet.entities['hashtags']:
                hashtags.append(tag['text'])
            popularity = tweet.favorite_count + tweet.retweet_count
            tweetObject = Tweet(user_id=tweet.id, text=tweet.text, hashtags=hashtags, user=user, popularity=popularity)
            tweetObjects.append(tweetObject)
    return tweetObjects


def convertDictTweetsToObjects(tweets, addUser=False):
    tweetObjects = []
    tweetsData = tweets['data']
    count = 0
    user = User()
    for tweet in tweetsData:
        if isGermanTweet(tweet):
            if addUser:
                u = api.getUser1(user_id=tweet['author_id'])
                user = convertUserToObject(u)
            hashtags = []
            try:
                for tag in tweet['entities']['hashtags']:
                    hashtags.append(tag['tag'])
            except:
                hashtags.append("")
            popularity = tweet['public_metrics']['like_count'] + tweet['public_metrics']['retweet_count']
            tweetObject = Tweet(user_id=tweet['id'], text=tweet['text'], hashtags=hashtags, user=user, popularity=popularity)
            tweetObjects.append(tweetObject)
        count += 1
    return tweetObjects, count


def convertUserToObject(user):
    userObject = User(user.id, user.screen_name, user.protected, user.followers_count,
                      user.verified)
    return userObject


def convertDictUserToObject(user):
    userData = user['data']
    try:
        userObject = User(user_id=userData['id'], name=userData['name'],
                          protected=userData['protected'],
                          followers=userData['public_metrics']['followers_count'],
                          verified=userData['verified'])
    except:
        userObject = User(user_id=userData['id'], name=userData['name'])
    return userObject
