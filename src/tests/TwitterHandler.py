import tweepy
import pandas as pd

client = tweepy.Client(
    consumer_key="YvEEBrW7pZiNTtUYn5XsSUYgg",
    consumer_secret="qCuZtrN0rjWMMkR6IcjyoqG6CZYSCZXFthrQyRofAE8DAg5flM",
    access_token="1504119098950705152-pxyiHB24q8gLZoGTFx7ParucCUy8mr",
    access_token_secret="hpKRZ5hm1aiQOkcAKB6hcUnVPLHotQxu0kP9lhq5wwLGR"
)

auth = tweepy.OAuthHandler(client.consumer_key, client.consumer_secret)
auth.set_access_token(client.access_token, client.access_token_secret)
api = tweepy.API(auth)

public_tweets = api.home_timeline()

for tweet in public_tweets:
    print(tweet.text)
