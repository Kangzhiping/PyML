from kafka.client import KafkaClient
from kafka import KafkaConsumer

class Consumer():
    '''
    使用Kafka—python的消费模块
    '''

    def __init__(self, kafkahost, kafkaport, kafkatopic, groupid):
        self.kafkaHost = kafkahost
        self.kafkaPort = kafkaport
        self.kafkatopic = kafkatopic
        self.groupid = groupid
        self.consumer = KafkaConsumer(self.kafkatopic, group_id = self.groupid,
                                      bootstrap_servers = '{kafka_host}:{kafka_port}'.format(
            kafka_host=self.kafkaHost,
            kafka_port=self.kafkaPort ))

    def consume_data(self):
        for message in self.consumer:
            yield message

if __name__ == '__main__':
    consumer = Consumer('node1', 9092, "testkafka", 'test-python-ranktest')
    # message 会延时加载，执行到下面的print 的时候去拉取记录，没有记录时候会等待kafka中有记录再执行。
    message = consumer.consume_data()
    for i in message:
        print(i.value)