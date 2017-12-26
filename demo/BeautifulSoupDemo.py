import urllib.request
from bs4 import BeautifulSoup

# 使用BeautifulSoup来解析
f=urllib.request.urlopen('http://www.baidu.com').read().lower()
try:
    s=f.decode('gbk')
except:
    s=f.decode('utf-8')
soup = BeautifulSoup(s)
print (soup.title)
newstext=soup.find(id="p_content")
ss=''
for item in str(newstext):
    if item in r'''0123456789{}%abcdefghijklmnopqrstuvwxyz'<>()?$+:&;|#!/"=-_.''':
        continue
    ss+=item
print (ss)