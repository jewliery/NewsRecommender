from models.Rating import Rating
from models.UserObject import User
from models.TweetObject import Tweet
from helper.TwitterManager import TwitterManager
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import spacy
import nltk

stop_words_de = ["co","http","https","a","ab","aber","ach","acht","achte","achten","achter","achtes","ag","alle","allein","allem","allen","aller","allerdings","alles","allgemeinen","als","also","am","an","andere","anderen","andern","anders","au","auch","auf","aus","ausser","außer","ausserdem","außerdem","b","bald","bei","beide","beiden","beim","beispiel","bekannt","bereits","besonders","besser","besten","bin","bis","bisher","bist","c","d","da","dabei","dadurch","dafür","dagegen","daher","dahin","dahinter","damals","damit","danach","daneben","dank","dann","daran","darauf","daraus","darf","darfst","darin","darüber","darum","darunter","das","dasein","daselbst","dass","daß","dasselbe","davon","davor","dazu","dazwischen","dein","deine","deinem","deiner","dem","dementsprechend","demgegenüber","demgemäss","demgemäß","demselben","demzufolge","den","denen","denn","denselben","der","deren","derjenige","derjenigen","dermassen","dermaßen","derselbe","derselben","des","deshalb","desselben","dessen","deswegen","d.h","dich","die","diejenige","diejenigen","dies","diese","dieselbe","dieselben","diesem","diesen","dieser","dieses","dir","doch","dort","drei","drin","dritte","dritten","dritter","drittes","du","durch","durchaus","dürfen","dürft","durfte","durften","e","eben","ebenso","ehrlich","ei","ei,","eigen","eigene","eigenen","eigener","eigenes","ein","einander","eine","einem","einen","einer","eines","einige","einigen","einiger","einiges","einmal","eins","elf","en","ende","endlich","entweder","er","Ernst","erst","erste","ersten","erster","erstes","es","etwa","etwas","euch","f","früher","fünf","fünfte","fünften","fünfter","fünftes","für","g","gab","ganz","ganze","ganzen","ganzer","ganzes","gar","gedurft","gegen","gegenüber","gehabt","gehen","geht","gekannt","gekonnt","gemacht","gemocht","gemusst","genug","gerade","gern","gesagt","geschweige","gewesen","gewollt","geworden","gibt","ging","gleich","gott","gross","groß","grosse","große","grossen","großen","grosser","großer","grosses","großes","gut","gute","guter","gutes","h","habe","haben","habt","hast","hat","hatte","hätte","hatten","hätten","heisst","her","heute","hier","hin","hinter","hoch","i","ich","ihm","ihn","ihnen","ihr","ihre","ihrem","ihren","ihrer","ihres","im","immer","in","indem","infolgedessen","ins","irgend","ist","j","ja","jahr","jahre","jahren","je","jede","jedem","jeden","jeder","jedermann","jedermanns","jedoch","jemand","jemandem","jemanden","jene","jenem","jenen","jener","jenes","jetzt","k","kam","kann","kannst","kaum","kein","keine","keinem","keinen","keiner","kleine","kleinen","kleiner","kleines","kommen","kommt","können","könnt","konnte","könnte","konnten","kurz","l","lang","lange","leicht","leide","lieber","los","m","machen","macht","machte","mag","magst","mahn","man","manche","manchem","manchen","mancher","manches","mann","mehr","mein","meine","meinem","meinen","meiner","meines","mensch","menschen","mich","mir","mit","mittel","mochte","möchte","mochten","mögen","möglich","mögt","morgen","muss","muß","müssen","musst","müsst","musste","mussten","n","na","nach","nachdem","nahm","natürlich","neben","nein","neue","neuen","neun","neunte","neunten","neunter","neuntes","nicht","nichts","nie","niemand","niemandem","niemanden","noch","nun","nur","o","ob","oben","oder","offen","oft","ohne","Ordnung","p","q","r","recht","rechte","rechten","rechter","rechtes","richtig","rund","s","sa","sache","sagt","sagte","sah","satt","schlecht","Schluss","schon","sechs","sechste","sechsten","sechster","sechstes","sehr","sei","seid","seien","sein","seine","seinem","seinen","seiner","seines","seit","seitdem","selbst","sich","sie","sieben","siebente","siebenten","siebenter","siebentes","sind","so","solang","solche","solchem","solchen","solcher","solches","soll","sollen","sollte","sollten","sondern","sonst","sowie","später","statt","t","tag","tage","tagen","tat","teil","tel","tritt","trotzdem","tun","u","über","überhaupt","übrigens","uhr","um","und","und?","uns","unser","unsere","unserer","unter","v","vergangenen","viel","viele","vielem","vielen","vielleicht","vier","vierte","vierten","vierter","viertes","vom","von","vor","w","wahr?","während","währenddem","währenddessen","wann","war","wäre","waren","wart","warum","was","wegen","weil","weit","weiter","weitere","weiteren","weiteres","welche","welchem","welchen","welcher","welches","wem","wen","wenig","wenige","weniger","weniges","wenigstens","wenn","wer","werde","werden","werdet","wessen","wie","wieder","will","willst","wir","wird","wirklich","wirst","wo","wohl","wollen","wollt","wollte","wollten","worden","wurde","würde","wurden","würden","x","y","z","z.b","zehn","zehnte","zehnten","zehnter","zehntes","zeit","zu","zuerst","zugleich","zum","zunächst","zur","zurück","zusammen","zwanzig","zwar","zwei","zweite","zweiten","zweiter","zweites","zwischen","zwölf","euer","eure","hattest","hattet","jedes","mußt","müßt","sollst","sollt","soweit","weshalb","wieso","woher","wohin"]


def convertTweetsToObjects(tweets):
    tweetObjects = []
    for tweet in tweets:
        user = User(tweet.user.id, tweet.user.screen_name, tweet.user.protected, tweet.user.followers_count,
                    tweet.user.verified)
        rating = Rating(tweet.favorited, tweet.retweeted, 0)
        hashtags = []
        for tag in tweet.entities['hashtags']:
            hashtags.append(tag['text'])
        tweetObject = Tweet(tweet.id, tweet.text, hashtags, user, False, tweet.lang, rating, False)
        tweetObjects.append(tweetObject)
    return tweetObjects

def convertDictTweetsToObjects(tweets):
    tweetObjects = []
    for tweet in tweets['data']:
        hashtags = []
        try:
            for tag in tweet['entities']['hashtags']:
                hashtags.append(tag['tag'])
        except:
            hashtags.append("")
        tweetObject = Tweet(id=tweet['id'], text=tweet['text'], hashtags=hashtags)
        tweetObjects.append(tweetObject)
    return tweetObjects

def normalization(texts):
    string = str(texts)
    de_nlp = spacy.load(('de_core_news_sm'))
    stemmer = nltk.stem.PorterStemmer()
    doc_spacy = de_nlp(string)
    print("Lemmatisierung:")
    print([token.lemma_ for token in doc_spacy])
    print("Stemming:")
    print([stemmer.stem(token.norm_.lower()) for token in doc_spacy])


def convertUserToObject(user):
    userObject = User(user.id, user.screen_name, user.protected, user.followers_count,
                      user.verified)
    return userObject

# Vectorize Text Data with Count Vectorizer
def vectorizeTexts(texts):
    vectorizer = CountVectorizer(stop_words=stop_words_de)
    X = vectorizer.fit_transform(texts)
    #print("Größe des Vokabulars: {}".format(len(vectorizer.vocabulary_)))
    #print("Inhalt des Vokabulars:\n {}".format(vectorizer.vocabulary_))
    features = vectorizer.get_feature_names_out()
    vector = X.toarray()
    #print("Merkmale:\n{}".format(features[:50]))
    return vector, features

# Vectorize Text Data with TfIdf
def tfIdfVectorizeTexts(texts):
    tfIdfVectorizer = TfidfVectorizer(use_idf=True)
    tfIdf = tfIdfVectorizer.fit_transform(texts)
    vector = tfIdf.toarray()
    return vector

# Vectorize Data with key value pairs
def vectorizeData(data):
    vectorizer = DictVectorizer()
    vector = vectorizer.fit_transform(data).toarray()
    return vector

# Creates Vector consisting of Text and Hashtags from Tweet
def createVectors(tweets):
    texts = []
    hashtags = []
    for tweet in tweets:
        texts.append(tweet.text)
        hashtags.append(tweet.hashtags)

    vectorTexts, featuresText = vectorizeTexts(texts)
    vectorHashtags, featuresHashtags = vectorizeTexts(hashtags)
    vectorHashtags = vectorHashtags * 4

    features = np.concatenate([featuresText, featuresHashtags], axis=0).tolist()
    vector = np.concatenate([vectorTexts, vectorHashtags], axis=1).tolist()
    return vector, features

def createVectorsTfIdf(tweets):
    texts = []
    hashtags = []
    for tweet in tweets:
        texts.append(tweet.text)
        hashtags.append(tweet.hashtags)

    vectorTexts = tfIdfVectorizeTexts(texts)
    vectorHashtags = tfIdfVectorizeTexts(hashtags)
    vectorHashtags = vectorHashtags * 4

    vector = np.concatenate([vectorTexts, vectorHashtags], axis=1).tolist()
    return vector


# Creates Vector consisting of Text, Hashtags and Meta Information from Tweet
# Careful: Tweet Object must contain a User!
def createFullVectors(tweets):
    texts = []
    hashtags = []
    data = []
    user = []
    for tweet in tweets:
        texts.append(tweet.text)
        hashtags.append(tweet.hashtags)
        data.append(tweet.keyValuePairs)
        user.append(tweet.user.keyValuePairs)

    vectorTexts, featuresText = vectorizeTexts(texts)
    vectorHashtags, featuresHashtags = vectorizeTexts(hashtags)
    vectorHashtags = vectorHashtags * 4
    vectorData = vectorizeData(data).astype(int)
    vectorUser = vectorizeData(user).astype(int)

    features = np.concatenate([featuresText, featuresHashtags], axis=0).tolist()
    vector = np.concatenate([vectorTexts, vectorHashtags, vectorData, vectorUser], axis=1).tolist()
    return vector, features

# Creates Vector consisting of Text and Hashtags from Tweet
def testNormalization(tweets):
    texts = []
    hashtags = []
    for tweet in tweets:
        texts.append(tweet.text)
        hashtags.append(tweet.hashtags)

    normalization(texts)
    normalization(hashtags)





