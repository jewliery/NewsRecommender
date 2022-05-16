import tweepy
from helper.MyStreamListener import MyStreamListener


class TwitterManager:
    BOUNDING_BOX_USA = -125.00,24.94,-66.93,49.59
    BOUNDING_BOX_DE = 6.17, 47.75, 14.49, 54.55
    BOUNDING_BOX_HAN = 9.543339, 52.265626, 9.877049, 52.440934

    consumer_key = "CuBKFmw0CivLjchfbHlYOUB2M"
    consumer_secret = "k3rqurXRGQqcv0WqB7Y1OicWdqoiIo9oKnqJ9DrpHiwuf3WG0m"
    access_token = "1504119098950705152-0yVcDDITfu7xwXwY1wiXaMCIRArbTw"
    access_token_secret = "4SHqywVzmJQCwxWRBlvNI0uOTYn2UqKos40RoQGVdYNHI"
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAE4zaQEAAAAAUUzTh37lC5EwwcvO2euJ%2FbQnfqU%3D2FS6WOjomaTTGpwGbpi78ezhtEAZQ44wa8bi5I5OJn8fwj6xrB"

    def __init__(self):
        self.client2 = tweepy.Client(bearer_token=self.bearer_token, return_type=dict)
        self.client = tweepy.Client(
            bearer_token=self.bearer_token,
            consumer_key=self.consumer_key,
            consumer_secret=self.consumer_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret
        )
        self.api = self.createConnection()

    def createConnection(self):
        auth = tweepy.OAuthHandler(self.client.consumer_key, self.client.consumer_secret)
        auth.set_access_token(self.client.access_token, self.client.access_token_secret)
        api = tweepy.API(auth)
        return api

    #------------------------------My Account------------------------------------
    def getTimeline(self):
        timeline_tweets = self.api.home_timeline()
        return timeline_tweets

    def getFavoriteTweets(self):
        favorite_tweets = self.api.get_favorites()
        return favorite_tweets

    # Returns Retweets from Statuses from logged in person
    def getRetweets(self):
        retweets = self.api.get_retweets_of_me()
        return retweets

    #-------------------------------Unabhängiges---------------------------------
    def getRandomTweets(self):
        ml = MyStreamListener(self.consumer_key,
                                self.consumer_secret,
                                self.access_token,
                                self.access_token_secret)
        stream = ml.createStream(self.consumer_key,
                                self.consumer_secret,
                                self.access_token,
                                self.access_token_secret)
        tweets = stream.filter(locations=self.BOUNDING_BOX_DE, threaded=True)
        return tweets

    def getTweets(self, query):
        tweets = self.api.search_tweets(q=query, lang="de", result_type="popular")
        return tweets

    # Returns Tweets of User with specified ID
    def getUserTimeline(self, userID):
        tweets = self.api.user_timeline(userID)
        return tweets


    #------------------------API V2----------------------------
    # Returns Liked Tweets of User with specified ID
    def getLikedTweets(self, id):
        tweets = self.client2.get_liked_tweets(id=id, tweet_fields=['author_id','public_metrics'])
        return tweets

    def getUser(self, name='', user_id=0):
        if name == '':
            user = self.client2.get_user(id=user_id, user_fields=['entities','protected','verified','public_metrics'])
        elif user_id == 0:
            user = self.client2.get_user(username=name,user_fields=['entities', 'protected', 'verified', 'public_metrics'])
        return user

    def getAllMyLikedTweets(self):
        tweets = self.client2.get_liked_tweets(id=1504119098950705152, tweet_fields=['entities','possibly_sensitive','author_id','lang'])
        return tweets

    def getAllFollowing(self, user_id):
        user = self.client2.get_users_following(id=user_id)
        return user

    def getAllUsersTweets(self, user_id):
        tweets = self.client2.get_users_tweets(id=user_id, max_results=5)
        return tweets



