#command: python tsla.py -k 192.168.1.14:9092,192.168.1.66:9093,192.168.1.66:9094 -t tsla-stock

import argparse
import decimal
import json
import csv
import io

from confluent_kafka import Producer


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Data generator for spark book")
    parser.add_argument("--kafka-brokers", "-k",
                        nargs="?",
                        help="Comma-separated list of kafka brokers. If specified, data is published to kafka.")
    parser.add_argument("--kafka-topic", "-t",
                        nargs="?",
                        help="Topic name to publish sensor data(default: tsla-stock).")

    args = parser.parse_args()

    p = Producer({'bootstrap.servers': args.kafka_brokers}) if args.kafka_brokers else None
    topic = args.kafka_topic or "tsla-stock"

    try:
        csvfile = io.open('/var/lib/kafka/TSLA-max', 'r', encoding='utf-8')
        reader = csv.DictReader(csvfile)
        for data in reader:
            out = json.dumps(data)
            print(out)
            if out is not None:
                p.produce(topic, value=out)
    except KeyboardInterrupt:
        pass
    finally:
        if p is not None:
            p.flush()