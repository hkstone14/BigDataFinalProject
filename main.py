import pandas as pd
import csv
from textblob import TextBlob
import re
import emoji
import nltk
from pyspark.ml.classification import DecisionTreeClassifier

nltk.download('words')
words = set(nltk.corpus.words.words())

infile = 'training.1600000.processed.noemoticon.csv'
clean_tweet = ''

training_data = pd.read_csv(infile, encoding="ISO-8859-1")
y = training_data['0']

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)


def cleaner(tweet):
    tweet = re.sub("@[A-Za-z0-9]+", "", tweet)
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet)
    tweet = " ".join(tweet.split())
    tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI)
    tweet = tweet.replace("#", "").replace("_", " ")
    tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
                     if w.lower() in words or not w.isalpha())
    return tweet


with open(infile, 'r') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        data = row[5]
        clean_tweet = cleaner(data)
        model = DecisionTreeClassifier()
        model.fit(clean_tweet, y)
