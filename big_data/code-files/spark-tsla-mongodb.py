#command: spark-submit --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.0.0,org.mongodb.spark:mongo-spark-connector_2.11:2.4.0,org.mongodb.mongo-hadoop:mongo-hadoop-core:1.3.1,org.mongodb:mongo-java-driver:3.1.0 --jars jars/mongo-hadoop-spark-2.0.2.jar tsla-mongodb.py 192.168.1.14:9092,192.168.1.14:9093,192.168.1.14:9094 tsla-stock

import sys
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import pymongo_spark
import json
import time

pymongo_spark.activate()

if __name__ == "__main__":
    conf = SparkConf().setAppName("tslaStockApp").setMaster("spark://192.168.1.6:7077")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("OFF")
    ssc = StreamingContext(sc, 5)
    brokers, topic = sys.argv[1:]
    kvs = KafkaUtils.createDirectStream(ssc, [topic],{"metadata.broker.list": brokers})
    lines = kvs.map(lambda x: json.loads(x[1]))

    def convert(date):
        epoch = float(time.mktime(time.strptime(date, '%Y-%m-%d')))
        return epoch

    rdd = lines.map(lambda x : {"Date": convert(x['Date']), "Open": x['Open'], "High":  float(x['High']), 
                                "Low": float(x['Low']), "Close": x['Close'], "Adj Close": x['Adj Close'], 
                                "Volume": x['Volume']})
    rdd.pprint()
    rdd.foreachRDD(lambda z: z.saveToMongoDB('mongodb://192.168.1.21:27017/tsla-stockdb.stockdata'))
    ssc.start()
    ssc.awaitTermination()
    