class Rating():
    def __init__(self, favorited=False, retweeted=False, klicks=0):
        self.favorited: bool = favorited
        self.retweeted: bool = retweeted
        self.klicks: int = klicks

    def getRanking(self):
        ranking: int = 0
        if self.klicks >= 1:
            ranking += 1
        if self.favorited:
            ranking += 2
        if self.retweeted:
            ranking += 3
        return ranking



