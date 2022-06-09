import json
#from helper.MyStreamListener import MyStreamListener
from models.Rating import Rating
from models.UserObject import User
from models.TweetObject import Tweet
from helper.TwitterManager import TwitterManager
from helper.FeatureSelection import *
from helper.DataHelper import *
from helper.Learner import *
from helper.Visualizer import *
from helper.DataPreprocessor import *
from helper.Modeling import *
from helper.Recommender import *

#api = TwitterManager()

#--------------------------Train Approaches----------------------------
# x_train, y_train = getOverallData()
# train2DData(x_train, y_train)
# trainOCSVM(x_train)

#----------------------------Visualization------------------------------
# tweetVectors, features = getAllMyLikedVectorData()
# tweetVectorsTfIdf = getAllLikedVectorDataTfIdf()
# showBarGraph(tweetVectors, features)
# showBar(tweetVectors, features)
# show2DVisualization(tweetVectors)

# tweets = api.getAllMyLikedTweets()
# tweetObjects = convertDictTweetsToObjects(tweets)
# showAnother2DVisualization(tweetObjects)

#--------------------------------Testing--------------------------------
# tweet_obj = getAllMyLikedRawData()
# for t in tweet_obj:
#     t.print()

#-----------------------------------Test----------------------------------

#testModels("jules3x")
clf = createUserModel("jules3x", "random-forest")



















