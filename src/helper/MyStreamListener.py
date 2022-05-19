# import tweepy
#
# class MyStreamListener(tweepy.Stream):
#
#     def on_status(self, status):
#         print(status.text)
#
#     def on_error(self, status_code):
#         if status_code == 420:  # end of monthly limit rate (500k)
#             return False
#
#     def createStream(self, consumer_key,
#                             consumer_secret,
#                             access_token,
#                             access_token_secret):
#         stream = MyStreamListener(consumer_key,
#                                   consumer_secret,
#                                   access_token,
#                                   access_token_secret)
#         return stream




