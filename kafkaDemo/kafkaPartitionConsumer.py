import time
from kafka.consumer import simple
import partition_consumer
import threading
from kafka.client import KafkaClient
#from kafka.consumer import SimpleConsumer
from kafka.consumer import KafkaConsumer

class Consumer(threading.Thread):
    daemon = True

    def __init__(self, partition_index):
        threading.Thread.__init__(self)
        self.part = [partition_index]
        self.__offset = 0

    def run(self):
        client = KafkaClient("node1:9092,node2:9092,node3:9092")
        consumer = KafkaConsumer(client, "test-group","testkafka", auto_commit=False,partitions=self.part)
        consumer.seek(0,0)
        while True:
            message = consumer.fetch_messages(True,60)
            self.__offset = message.offset
            print(message.message.value)

if __name__ == '__main__':
    threads = []
    partition = 3
    for index in range(partition):
        threads.append(partition_consumer.Consumer(index))

    for t in threads:
        t.start()

    time.sleep(50000)