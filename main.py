import pandas as pd
import csv

from pyspark.ml.feature import StringIndexer
from textblob import TextBlob
import re
import emoji
import nltk
from pyspark.ml.classification import DecisionTreeClassifier

nltk.download('words')
words = set(nltk.corpus.words.words())

infile = 'training.1600000.processed.noemoticon.csv'
clean_tweet = ''

# y param for training model
training_data = pd.read_csv(infile, encoding="ISO-8859-1")
y = training_data['0']


def cleaner(tweet):
    tweet = re.sub("@[A-Za-z0-9]+", "", tweet)
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet)
    tweet = " ".join(tweet.split())
    tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI)
    tweet = tweet.replace("#", "").replace("_", " ")
    # tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
    #                  if w.lower() in words or not w.isalpha())
    return tweet


# x(clean_tweet) for training model
with open(infile, 'r') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        data = row[5]
        clean_tweet = cleaner(data)
        X = pd.DataFrame(eval(clean_tweet))
        model = DecisionTreeClassifier()
        model.fit(X, y)
