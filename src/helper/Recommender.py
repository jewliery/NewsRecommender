from helper.Modeling import *
from helper.DataPreprocessor import UserData

# Nimmt die Testdaten und zeigt welche Tweets empfohlen wurden
def getRecommendationList(pred, test, userData):
    recommendVectors = []
    # Wenn 1 ist suche tweet und empfehle diesen
    for i in range(0, len(pred)):
        if pred[i] == 1:
            recommendVectors.append(test[i])

    recommend = []
    for i in range(0, len(userData.x_train)):
        for r in recommendVectors:
            if userData.x_train[i] == r:
                recommend.append(userData.train[i])
                userData.train[i].print()

    return recommend

