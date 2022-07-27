from helper.Visualizer import *
from helper.Recommender import *

userData = UserData(user_name="jules3x")
# ---------------------Visualization----------------------
show2DVisualization(userData.train)
showAnother2DVisualization(userData.train)


# -----------------------Evaluation-------------------------

# ----------PLAIN RECOMMENDER---------
# createUserModel(userData, "naive-bayes")
# showResult("plain")

# -------BOUNDED-GREEDY-SELECTION--------
# boundedGreedySelection(userData, 10)
# showResult("bgs")

# ------USER PROFILE PARTITIONING------
# profile_partitioning(userData, 10)
# showResult("upp")

# ------ANOMALIES AND EXCEPTIONS------
# rec = anomaliesExceptions(userData, 10)
# showResult("aua")

# --------EVERYTHING AT ONCE--------
testModels(userData)
showEvaluation()

