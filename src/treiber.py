from helper.Visualizer import *
from helper.Recommender import *

# ---------------------GET DATA----------------------------
userData = UserData(user_name="jules3x")
getTrainingData(userData)

# ---------------------Visualization----------------------
# show2DVisualization(userData.x_train)

# -----------------------Evaluation-------------------------

# -----TEST DIFFERENT CLASSIFIER------
# testModels(userData)

# ----------PLAIN RECOMMENDER---------
# createUserModel(userData, "naive-bayes")
# showResult("plain")

# -------BOUNDED-GREEDY-SELECTION--------
# boundedGreedySelection(userData, 10)
# showResult("bgs")

# ------USER PROFILE PARTITIONING------
profile_partitioning(userData, 10)
showResult("upp")

# ------ANOMALIES AND EXCEPTIONS------
# anomaliesExceptions(userData, 10)
# showResult("aua")

# ---------TEST EVERY METHOD-----------
# createUserModel(userData, "naive-bayes")
# boundedGreedySelection(userData, 10)
# profile_partitioning(userData, 10)
# anomaliesExceptions(userData, 10)
# showEvaluation()
