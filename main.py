from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
from textblob import TextBlob
import findspark as fs
from pyspark.context import SparkContext
from pyspark import SQLContext, SparkConf


def cleaner(tweets):
    words = tweets.select(explode(split(tweets.value, "t_end")).alias("word"))
    words = words.na.replace('', None)
    words = words.na.drop()
    words = words.withColumn('word', F.regexp_replace('word', r'http\S+', ''))
    words = words.withColumn('word', F.regexp_replace('word', '@\w+', ''))
    words = words.withColumn('word', F.regexp_replace('word', '#', ''))
    words = words.withColumn('word', F.regexp_replace('word', 'RT', ''))
    words = words.withColumn('word', F.regexp_replace('word', ':', ''))
    return words


# text classification
def polarity_detection(text):
    return TextBlob(text).sentiment.polarity


def subjectivity_detection(text):
    return TextBlob(text).sentiment.subjectivity


def text_classification(words):
    # polarity detection
    polarity_detection_udf = udf(polarity_detection, StringType())
    words = words.withColumn("polarity", polarity_detection_udf("word"))
    # subjectivity detection
    subjectivity_detection_udf = udf(subjectivity_detection, StringType())
    words = words.withColumn("subjectivity", subjectivity_detection_udf("word"))
    return words


if __name__ == "__main__":
    # create Spark session
    fs.init()
    spark = SparkSession.builder.appName("MyBigDataStreamingApp").getOrCreate()
    # read the tweet data from socket
    tweets = spark.readStream.format("socket").option("host", "0.0.0.0").option("port", 5555).load()
    # Preprocess the data
    words = cleaner(tweets)
    # text classification to define polarity and subjectivity
    words = text_classification(words)
    words = words.repartition(1)
    print(words)
