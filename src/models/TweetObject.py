# TweetObject, this class will contain a Tweet
import string
from models.Rating import Rating


class Tweet:

    def __init__(self, id=0, text='', hashtags=(), user = None, popular=False, language='en', favorited=False, retweeted=False, klick=0, sensitive=False):
        self.id: int = id
        self.text: string = text
        self.hashtags: list = hashtags
        self.user: User = user
        self.popular: bool = popular
        self.language: string = language
        self.rating: Rating = Rating(favorited, retweeted, klick)
        self.sensitive: bool = sensitive
        self.vector: list = self.createVector()

    def createVector(self) -> list:
        vector = ()
        return vector

    def createWordVector(self) -> list:
        vector = ()
        return vector

    def favoriteTweet(self):
        self.rating = Rating(True, self.rating.retweeted, self.rating.klicks)

    def retweetTweet(self):
        self.rating = Rating(self.rating.favorited, True, self.rating.klicks)

    def klickTweet(self):
        self.rating = Rating(self.rating.favorited, self.rating.retweeted, self.rating.klicks + 1)

