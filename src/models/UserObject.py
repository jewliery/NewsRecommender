class User:

    def __init__(self, id=0, name='', protected=False, followers=0, verified=False, language='en'):
        self.id = id
        self.name = name
        self.protected = protected
        self.followers = followers
        self.verified = verified
        self.language = language

        self.keyValuePairs = self.createKeyValuePairs()
        self.favorite_tweets = []

    def createKeyValuePairs(self) -> list:
        vector = {'protected': self.protected, 'followers': self.followers,
                  'verified': self.verified, 'language': self.language}
        return vector

    def addFavoriteTweet(self, tweet):
        self.favorite_tweets.append(tweet)



