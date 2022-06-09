from helper.Modeling import *
from helper.DataPreprocessor import UserData


def getRecommendationList(clf, userData):
    data = userData.getData()
    recommendation = clf.predict(data)
    return recommendation
