# TweetObject, this class will contain a Tweet
import string
from models.Rating import Rating


def createHashtagString(hashtags):
    string = ''
    for tag in hashtags:
        string += tag + ' '
    return string

class Tweet:

    def __init__(self, id=0, text='', hashtags=[], user=None, popular=False, language='de', favorited=False,
                 retweeted=False, klick=0, sensitive=False):
        self.id: int = id
        self.text: string = text
        self.hashtags = createHashtagString(hashtags)
        self.user: User = user
        self.popular: bool = popular
        self.language: string = language
        self.rating: Rating = Rating(favorited, retweeted, klick)
        self.sensitive: bool = sensitive
        self.keyValuePairs: list = self.createKeyValuePairs()

    def __hash__(self):
        return hash(self.text)

    def __eq__(self, other):
        return self.id == other.id

    def __cmp__(self, other):
        return self.id == other.id

    def  __gt__(self, other):
        return self.id > other.id

    def createKeyValuePairs(self) -> list:
        vector = {'popular': self.popular, 'language': self.language}
        return vector


    def favoriteTweet(self):
        self.rating = Rating(True, self.rating.retweeted, self.rating.klicks)

    def retweetTweet(self):
        self.rating = Rating(self.rating.favorited, True, self.rating.klicks)

    def klickTweet(self):
        self.rating = Rating(self.rating.favorited, self.rating.retweeted, self.rating.klicks + 1)

    def print(self, showUser=True):
        print('')
        print('------TWEET INFO------')
        print('Id: ' + str(self.id))
        print('Text: ' + self.text)
        print('Hashtags: ' + self.hashtags)
        print('Popular: ' + str(self.popular))
        print('Language: ' + self.language)
        rating = 'Rating: '
        if self.rating.favorited:
            rating += 'favorited '
        if self.rating.retweeted:
            rating += 'retweeted '
        print(rating)

        if showUser & (self.user is not None):
            print('')
            print('------USER INFO------')
            print('Id: ' + str(self.user.id))
            print('Username: ' + self.user.name)
            print('Protected: ' + str(self.user.protected))
            print('Followers: ' + str(self.user.followers))
            print('Verified: ' + str(self.user.verified))


