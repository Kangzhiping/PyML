import re
import urllib.request
import urllib

from collections import deque

queue = deque()
visited = set()

#url = 'http://baike.baidu.com/item/%E5%A4%A7%E5%AD%A6%E4%B8%93%E4%B8%9A?sefr=enterbtn'
url1 = 'http://baike.baidu.com/item/'
url2 = '%E5%A4%A7%E5%AD%A6%E4%B8%93%E4%B8%9A?sefr=enterbtn'
url = url1 + url2

queue.append(url)
cnt = 0

while queue:
    url = queue.popleft()  # 队首元素出队
    visited |= {url}  # 标记为已访问

    print('已经抓取: ' + str(cnt) + '   正在抓取 <---  ' + url)
    cnt += 1
    urlop = urllib.request.urlopen(url,timeout=20)
    if 'html' not in urlop.getheader('Content-Type'):
        continue

    # 避免程序异常中止, 用try..catch处理异常
    #try:
    #    data = urlop.read().decode('utf-8')
    #except:
    #    continue

    # 正则表达式提取页面中所有队列, 并判断是否已经访问过, 然后加入待爬队列
    # linkre = re.compile('href=\"(.+?)\"')
    # for x in linkre.findall(data):
    #     if 'http://' in x and x not in visited:
    #         queue.append(x)
    #         print('加入队列 --->  ' + x)
    # linkWord = re.compile('Python\"(.+?)\"')
    # for x in linkWord.findall(data):
    #     print('找出 --->  ' + x)
    # print (data)
    #words = re.compile('label-module=\"para\">(.+)</div>')
    # print (type(words.findall(data)))
    #for x in words.findall(data):
    #    if (x != ""):
    #        print(x)

    #获取所有中文
    data = urlop.read()
    try:
        s = data.lower().decode('gbk');
    except:
        s = data.lower().decode('utf-8');
    zs = ''
    for item in s:
        if item == ' ':
            continue
        if item in r'''0123456789{}[]%abcdefghijklmnopqrstuvwxyz'<>()?$+:&@;|*#!\/"=-_.''':
            continue
        zs += item
    print (zs)

