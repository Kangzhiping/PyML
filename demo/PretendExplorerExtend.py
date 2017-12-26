import urllib.request
import http.cookiejar
import re
import urllib.request
import urllib
from collections import deque

# head: dict of header
def makeMyOpener(head={
    'Connection': 'Keep-Alive',
    'Accept': 'text/html, application/xhtml+xml, */*',
    'Accept-Language': 'en-US,en;q=0.8,zh-Hans-CN;q=0.5,zh-Hans;q=0.3',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko'
}):
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    header = []
    for key, value in head.items():
        elem = (key, value)
        header.append(elem)
    opener.addheaders = header
    return opener

#def saveFile(data):
#    save_path = 'C:\\Users\\IBM_ADMIN\\MikeFile\\Python\\WebSpider.txt'
#    f_obj = open(save_path, 'wb') # wb 表示打开方式
#    f_obj.write(data)
#    f_obj.close()


queue = deque()
visited = set()
url = 'http://www.baidu.com/'
queue.append(url)
cnt = 0

save_path = 'C:\\Users\\IBM_ADMIN\\MikeFile\\Python\\WebSpider.txt'
f_obj = open(save_path, 'w') # wb 表示打开方式
f_obj.write('抓取以下链接：')

while queue:
    url_get = queue.popleft()  # 队首元素出队
    visited |= {url_get}  # 标记为已访问
    f_obj.write(url_get)
    oper = makeMyOpener()
    uop = oper.open(url_get, timeout=2)
    print('已经抓取: ' + str(cnt) + '   正在抓取 <---  ' + url_get)
    cnt += 1

    if 'html' not in uop.getheader('Content-Type'):
        continue

    # 避免程序异常中止, 用try..catch处理异常
    try:
        data = uop.read().decode('utf-8')
    except:
        continue

    # 正则表达式提取页面中所有队列, 并判断是否已经访问过, 然后加入待爬队列
    linkre = re.compile('href=\"(.+?)\"')
    for x in linkre.findall(data):
        if 'http://' in x and x not in visited:
            queue.append(x)
            print('加入队列 --->  ' + x)
f_obj.flush()
f_obj.close()