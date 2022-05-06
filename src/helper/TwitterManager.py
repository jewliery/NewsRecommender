import tweepy


class TwitterManager:

    def __init__(self):
        self.bearer_token = "AAAAAAAAAAAAAAAAAAAAAE4zaQEAAAAAUUzTh37lC5EwwcvO2euJ%2FbQnfqU%3D2FS6WOjomaTTGpwGbpi78ezhtEAZQ44wa8bi5I5OJn8fwj6xrB"
        self.client2 = tweepy.Client(bearer_token=self.bearer_token, return_type=dict)
        self.client = tweepy.Client(
            bearer_token=self.bearer_token,
            consumer_key="CuBKFmw0CivLjchfbHlYOUB2M",
            consumer_secret="k3rqurXRGQqcv0WqB7Y1OicWdqoiIo9oKnqJ9DrpHiwuf3WG0m",
            access_token="1504119098950705152-0yVcDDITfu7xwXwY1wiXaMCIRArbTw",
            access_token_secret="4SHqywVzmJQCwxWRBlvNI0uOTYn2UqKos40RoQGVdYNHI"
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

    #Returns Retweets from Statuses from logged in person
    def getRetweets(self):
        retweets = self.api.get_retweets_of_me()
        return retweets

    def getUserTimeline(self, userID):
        tweets = self.api.user_timeline(userID)
        return tweets

    def getTweets(self, query):
        tweets = self.api.search_tweets(q=query, lang="de", result_type="popular")
        return tweets

    def getBookmarks(self):
        tweets = self.client2.get_bookmarks()
        return tweets

    def getLikedTweets(self, id):
        tweets = self.client2.get_liked_tweets(id=id)
        return tweets

    def getUser(self, name):
        user = self.client2.get_user(username=name)
        return user

    def getAllMyLikedTweets(self):
        tweets = self.client2.get_liked_tweets(id=1504119098950705152, tweet_fields=['entities','possibly_sensitive','author_id','lang'])
        return tweets

