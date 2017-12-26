# python 的队列
from collections import deque
queue = deque(["Eric", "John", "Michael"])
queue.append("mike")
queue.append("dina")
queue.popleft()
queue.popleft()
queue.pop()
print(queue)