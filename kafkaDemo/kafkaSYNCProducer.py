from kafka import KafkaProducer
from kafka.errors import kafka_errors
import json

class Producer():

    def __init__(self, kafkahost, kafkaport, kafkatopic):
        self.kafkaHost = kafkahost
        self.kafkaPort = kafkaport
        self.kafkatopic = kafkatopic
        self.producer = KafkaProducer(bootstrap_servers = '{kafka_host}:{kafka_port}'.format(
            kafka_host=self.kafkaHost,
            kafka_port=self.kafkaPort))

    def Producer_send(self, params):
        try:
            parmas_message = json.dumps(params)
            producer = self.producer
            producer.send(self.kafkatopic,parmas_message.encode('utf-8'))
            producer.flush()
        except kafka_errors as e:
            print(e)

if __name__ == '__main__':
    producer = Producer('node1', 9092, "testkafka")
    for i in range(100):
        producer.Producer_send("producer Mike %s" %i)
