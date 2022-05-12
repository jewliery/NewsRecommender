import json

from models.Rating import Rating
from models.UserObject import User
from models.TweetObject import Tweet
from helper.TwitterManager import TwitterManager
from helper.MyStreamListener import MyStreamListener
from helper.FeatureSelection import *
from helper.DataHelper import *
from helper.Learner import *
from helper.Visualizer import *

api = TwitterManager()

#--------------------------Train Approaches----------------------------
x_train, y_train = getOverallData()
train2DData(x_train, y_train)
# trainOCSVM(x_train)

#----------------------------Visualization------------------------------
# tweetVectors, features = getAllTrainingsData()
# tweetVectorsTfIdf = getTfIdfAllTrainingsData()
# showBarGraph(tweetVectors, features)
# showBar(tweetVectors, features)
# show2DVisualization(tweetVectors)

# tweets = api.getAllMyLikedTweets()
# tweetObjects = convertDictTweetsToObjects(tweets)
# showAnother2DVisualization(tweetObjects)

#--------------------------------Testing--------------------------------
# tweets = api.getAllMyLikedTweets()
# tweet_obj = convertDictTweetsToObjects(tweets)
# for t in tweet_obj:
#     t.print()


#-----------------------------------Test----------------------------------


















