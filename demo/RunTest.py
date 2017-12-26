url = 'http://www.zhihu.com/'
opener = getOpener(header)
op = opener.open(url)
data = op.read()
data = ungzip(data)  # 解压
_xsrf = getXSRF(data.decode())

url += 'login'
id = '这里填你的知乎帐号'
password = '这里填你的知乎密码'
postDict = {
    '_xsrf': _xsrf,
    'email': id,
    'password': password,
    'rememberme': 'y'
}
postData = urllib.parse.urlencode(postDict).encode()
op = opener.open(url, postData)
data = op.read()
data = ungzip(data)

print(data.decode())  # 你可以根据你的喜欢来处理抓取回来的数据了!