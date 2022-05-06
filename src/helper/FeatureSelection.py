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

# Vectorize Text Data with Count Vectorizer
def vectorizeTexts(texts):
    vectorizer = CountVectorizer(stop_words=["https"])
    X = vectorizer.fit_transform(texts)
    features = vectorizer.get_feature_names_out()
    vector = X.toarray()
    return vector, features

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

# Creates Vector consisting of Text and Hashtags from Tweet
def createVectors(tweets):
    texts = []
    hashtags = []
    for tweet in tweets:
        texts.append(tweet.text)
        hashtags.append(tweet.hashtags)

    vectorTexts, featuresText = vectorizeTexts(texts)
    vectorHashtags, featuresHashtags = vectorizeTexts(hashtags)
    vectorHashtags = vectorHashtags * 4

    features = np.concatenate([featuresText, featuresHashtags], axis=0).tolist()
    vector = np.concatenate([vectorTexts, vectorHashtags], axis=1).tolist()
    return vector, features

def createVectorsTfIdf(tweets):
    texts = []
    hashtags = []
    for tweet in tweets:
        texts.append(tweet.text)
        hashtags.append(tweet.hashtags)

    vectorTexts = tfIdfVectorizeTexts(texts)
    vectorHashtags = tfIdfVectorizeTexts(hashtags)
    vectorHashtags = vectorHashtags * 4

    vector = np.concatenate([vectorTexts, vectorHashtags], axis=1).tolist()
    return vector


# Creates Vector consisting of Text, Hashtags and Meta Information from Tweet
# Careful: Tweet Object must contain a User!
def createFullVectors(tweets):
    texts = []
    hashtags = []
    data = []
    user = []
    for tweet in tweets:
        texts.append(tweet.text)
        hashtags.append(tweet.hashtags)
        data.append(tweet.keyValuePairs)
        user.append(tweet.user.keyValuePairs)

    vectorTexts, featuresText = vectorizeTexts(texts)
    vectorHashtags, featuresHashtags = vectorizeTexts(hashtags)
    vectorHashtags = vectorHashtags * 4
    vectorData = vectorizeData(data).astype(int)
    vectorUser = vectorizeData(user).astype(int)

    features = np.concatenate([featuresText, featuresHashtags], axis=0).tolist()
    vector = np.concatenate([vectorTexts, vectorHashtags, vectorData, vectorUser], axis=1).tolist()
    return vector, features





