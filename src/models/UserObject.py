from data.TwitterManager import TwitterManager


class User:

    def __init__(self, user_id=0, name='', protected=False, followers=0, verified=False):
        self.id = user_id
        self.name = name
        self.protected = protected
        self.followers = followers
        self.verified = verified

        self.following = []
        self.followersTweets = []

        self.keyValuePairs = self.createKeyValuePairs()
        self.favorite_tweets = []
        self.api = TwitterManager()

    def createKeyValuePairs(self):
        vector = {'name': self.name, 'protected': self.protected, 'followers': self.followers,
                  'verified': self.verified}
        return vector

    def addFavoriteTweet(self, tweet):
        self.favorite_tweets.append(tweet)

    def updateFollowingList(self):
        self.following = self.api.getAllFollowing(self.id)['data']


    def __hash__(self):
        return hash(str(self.name))

    def __eq__(self, other):
        return self.id == other.id

    def __cmp__(self, other):
        return self.id == other.id
