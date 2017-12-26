# get data from baidu with input data
import urllib
import urllib.request

data = {}
data['word'] = "信息管理与信息系统"
url_value = urllib.parse.urlencode(data)
url = 'http://www.baidu.com/s?'
full_url = url + url_value
print (full_url)

data = urllib.request.urlopen(full_url).read()
data = data.decode('UTF-8')
#print(data)

zs = ''
for item in data:
    if item == ' ':
        continue
    if item in r'''0123456789{}[]%abcdefghijklmnopqrstuvwxyz'<>()?$+:&@;|*#!\/"=-_.''':
        continue
    zs += item
print (zs)

