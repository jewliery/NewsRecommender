class User:

    def __init__(self):
        self.favorite_tweets: list = ()

    def addFavoriteTweet(self, tweet):
        self.favorite_tweets.insert(tweet)



