import tweepy


class TwitterManager:

    def __init__(self):
        self.client = tweepy.Client(
            consumer_key="YvEEBrW7pZiNTtUYn5XsSUYgg",
            consumer_secret="qCuZtrN0rjWMMkR6IcjyoqG6CZYSCZXFthrQyRofAE8DAg5flM",
            access_token="1504119098950705152-pxyiHB24q8gLZoGTFx7ParucCUy8mr",
            access_token_secret="hpKRZ5hm1aiQOkcAKB6hcUnVPLHotQxu0kP9lhq5wwLGR"
        )
        self.api = self.createConnection()

    def createConnection(self):
        auth = tweepy.OAuthHandler(self.client.consumer_key, self.client.consumer_secret)
        auth.set_access_token(self.client.access_token, self.client.access_token_secret)
        api = tweepy.API(auth)
        return api

    def getTimeline(self):
        timeline_tweets = self.api.home_timeline()
        return timeline_tweets

    def getRandomTweets(self):
        return

    def getFavoriteTweets(self):
        favorite_tweets = self.api.get_favorites()
        return favorite_tweets

    def getRetweets(self):
        retweets = self.api.get_retweets_of_me()
        return retweets

