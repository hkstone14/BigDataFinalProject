from ctypes import Array
import pandas as pd
import csv
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.util import MLUtils
# from textblob import TextBlob
import re

# import nltk
#
#
# nltk.download('words')
# words = set(nltk.corpus.words.words())

infile = 'training.1600000.processed.noemoticon.csv'
clean_tweet = ''

# y param for training model
training_data = pd.DataFrame(pd.read_csv(infile, encoding="ISO-8859-1"))
test_data = pd.DataFrame(pd.read_csv('testdata.manual.2009.06.14.csv', encoding="ISO-8859-1"))
# y = training_data['0']


def cleaner(tweet):
    tweet = re.sub("@[A-Za-z0-9]+", "", tweet)
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet)
    tweet = " ".join(tweet.split())
    # tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI)
    tweet = tweet.replace("#", "").replace("_", " ")
    # tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
    #                  if w.lower() in words or not w.isalpha())
    return tweet


# x(clean_tweet) for training model
# with open(infile, 'r') as csvfile:
if __name__ == "__main__":
    # rows = csv.reader(csvfile)
    # for row in rows:
    #     data = row[5]
    #     clean_tweet = cleaner(data)
    #     X = pd.DataFrame(eval(clean_tweet))

    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(training_data)
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(
        training_data)
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])
    model = pipeline.fit(training_data)
    predictions = model.transform(test_data)
    predictions.select("predictedLabel", "label", "features").show(5)
