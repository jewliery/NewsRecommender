import string
from models.UserObject import User


def createHashtagString(hashtags):
    s = ''
    for tag in hashtags:
        s += tag + ' '
    return s


class Tweet:

    def __init__(self, user_id=0, text='', hashtags=[], user=None, popularity=0):
        self.id: int = user_id
        self.text: string = text
        self.hashtags = createHashtagString(hashtags)
        self.user: User = user
        self.popularity = popularity

    def __hash__(self):
        return hash(self.text)

    def __eq__(self, other):
        return self.id == other.id

    def __cmp__(self, other):
        return self.id == other.id

    def __gt__(self, other):
        return self.id > other.id

    def createKeyValuePairs(self):
        vector = {'popular': self.popularity}
        return vector

    def print(self, showUser=True):
        print('')
        print('------TWEET INFO------')
        print('Id: ' + str(self.id))
        print('Text: ' + self.text)
        print('Hashtags: ' + self.hashtags)
        print('Popular: ' + str(self.popularity))

        if showUser & (self.user is not None):
            print('')
            print('------USER INFO------')
            print('Id: ' + str(self.user.id))
            print('Username: ' + self.user.name)
            print('Protected: ' + str(self.user.protected))
            print('Followers: ' + str(self.user.followers))
            print('Verified: ' + str(self.user.verified))
