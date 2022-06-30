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
from helper.DataPreprocessor import UserData
from helper.Modeling import *
from helper.Recommender import *

#api = TwitterManager()

#----------------------------Visualization------------------------------
# tweetVectors, features = getAllMyLikedTextVectorData()
#showBarGraph(tweetVectors, features)
# showBar(tweetVectors, features)
#show2DVisualization(tweetVectors)

# tweets = api.getAllMyLikedTweets()
# tweetObjects = convertDictTweetsToObjects(tweets)
# showAnother2DVisualization(tweetObjects)

#--------------------------------Testing--------------------------------
# tweet_obj = getAllMyLikedRawData()
# for t in tweet_obj:
#     t.print()

#-----------------------------------Test----------------------------------

#testModels("jules3x")
#clf = createUserModel("jules3x", "random-forest")

userData = UserData(user_name="jules3x")
#clf, pred, test = createUserModel(userData, "random-forest")
clf, pred, test = createUserModel(userData, "naive-bayes")
#boundedGreedySelection(pred, test, userData, 5)
#profilePartitioning(userData)


























