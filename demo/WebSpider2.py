import urllib.request
import urllib
from bs4 import BeautifulSoup
import pymysql
#import sys
#import re
#import os
#import os.path
#import pickle

#type = sys.getfilesystemencoding()
filePath = 'C:\\Users\\IBM_ADMIN\\Desktop\\在线教育\\Professional\\'
filePath1 = 'C:\\Users\\IBM_ADMIN\\Desktop\\在线教育\\Professional_html\\'

#百度百科为爬虫数据库
url1 = 'http://baike.baidu.com/item/'

#连接mysql
conn= pymysql.connect(
        host='localhost',
        port = 3306,
        user='mike',
        passwd='fendou01',
        db ='edtdoor',
        charset = 'utf8'
        )
cur = conn.cursor()
#cur.execute("update `Professional_master` set `ID` = '050307T', `Name` = '数字出版' where `Name` = ' '")
#list = cur.execute("SELECT * FROM `Professional_master` where `Class_name` = '新闻传播学类';")
#从数据库中提取所有的专业名称
list = cur.execute("SELECT `Name`, `ID` FROM `Professional_master`;")
#Professional_list = cur.fetchmany(list)
Professional_list = cur.fetchall()
print(Professional_list)

cur.close()
conn.commit()
conn.close()

def write_file(filename, contect):
    #f1 = open(filename,'wb')
    f1 = open(filename, 'w',encoding='utf-8')
    f1.write(contect)
    #pickle.dump(contect,f1) #将对象用流的方式写进file.
    f1.flush()
    f1.close()
    print("test01")

count = 1
for professional in Professional_list:
    url2 = professional[0] + '专业'
    # 中文搜索转化
    url = url1 + urllib.parse.quote(url2)

    urlop = urllib.request.urlopen(url,timeout=500)
    if 'html' not in urlop.getheader('Content-Type'):
        continue

    data = urlop.read()

    """
    try:
        dc = data.lower().decode('utf-8').encode(type);
    except:
        dc = data.lower().decode('gbk').encode(type);
    """
    soup= BeautifulSoup(data,"html.parser")
    #soup = BeautifulSoup(dc, "html.parser")

    contents = soup.find(class_="main-content")

    if contents is None:
        print(url2 + "没有找到main-content")
        print(data)
        #continue
        #如果找不到，去掉专业二个字
        url = url1 + urllib.parse.quote(professional[0])
        urlop = urllib.request.urlopen(url, timeout=500)
        if 'html' not in urlop.getheader('Content-Type'):
            continue

        data = urlop.read()

        soup = BeautifulSoup(data, "html.parser")
        print("正在爬取第 " + str(count) + "个--->>" + soup.title.string)

        contents = soup.find(class_="main-content")
        if contents is None:
            print(professional[0] + "没有找到main-content")
            continue

    print("正在爬取第 " + str(count) +  "个--->>" + soup.title.string )
    count += 1

    # class 是关键字，所以加一个下划线
    values = soup.findAll(class_="para")

    text = ''
    for value in values:
        if value.text == None:
            continue
        text = text + value.text + '\n'

    """
    zs = ''
    for item in dc:
        if item == ' ':
            continue
        if item in r'''0123456789{}[]%abcdefghijklmnopqrstuvwxyz'<>()?$+:&@;|*#!\/"=-_,./r/n/t''':
            continue
        zs += item
    """
    filename = filePath + url2 + '.txt'
    write_file(filename,text)

    #去掉不用的html标签
    if contents.find(class_="top-tool") is not None:
        contents.find(class_="top-tool").decompose()
    if contents.find(id="open-tag") is not None:
        contents.find(id="open-tag").decompose()
    val_edit = contents.findAll(class_="edit-lemma")
    for edit in val_edit:
        if edit is not None:
            edit.decompose()
    if contents.find(class_="lock-lemma") is not None:
        contents.find(class_="lock-lemma").decompose()
    #删除部分超链接
    val_href = contents.findAll('a')
    for href in val_href:
        if href.get('href') is None:
            continue
        if not str(href.get('href')).startswith("#"):
            del href['href']

    filename1 = filePath1 + professional[1] + '.html'
    write_file(filename1,str(contents))