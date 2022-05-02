from models.Rating import Rating
from models.UserObject import User
from models.TweetObject import Tweet
from helper.TwitterManager import TwitterManager
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
import numpy as np


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


def convertUserToObject(user):
    userObject = User(user.id, user.screen_name, user.protected, user.followers_count,
                      user.verified)
    return userObject

# Vectorize Text Data with Count Vectorizer
def vectorizeTexts(texts):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    vectorizer.get_feature_names_out()
    vector = X.toarray()
    return vector

# Vectorize Text Data with TfIdf
def tfIdfVectorizeTexts(texts):
    tfIdfVectorizer = TfidfVectorizer(use_idf=True)
    tfIdf = tfIdfVectorizer.fit_transform(texts)
    vector = tfIdf.toarray()
    return vector

# Vectorize Data with key value pairs
def vectorizeData(data):
    vectorizer = DictVectorizer()
    vector = vectorizer.fit_transform(data).toarray()
    return vector

def createVectors(tweets):
    texts = []
    hashtags = []
    data = []
    user = []
    for tweet in tweets:
        texts.append(tweet.text)
        hashtags.append(tweet.hashtags)
        data.append(tweet.keyValuePairs)
        user.append(tweet.user.keyValuePairs)

    vectorTexts = vectorizeTexts(texts)
    vectorHashtags = vectorizeTexts(hashtags)
    vectorData = vectorizeData(data).astype(int)
    vectorUser = vectorizeData(user).astype(int)

    vector = np.concatenate([vectorTexts, vectorHashtags, vectorData, vectorUser], axis=1).tolist()
    return vector





