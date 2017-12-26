import http.cookiejar
import urllib.request

# 处理cookiejar
# getOpener函数接收一个 head 参数, 这个参数是一个字典.函数把字典转换成元组集合, 放进opener.
# 这样我们建立的这个 opener 就有两大功能:
# 1. 自动处理使用opener过程中遇到的 Cookies
# 2. 自动在发出的GET或者POST请求中加上自定义的Header

def getOpener(head):
    # deal with the Cookies
    cj = http.cookiejar.CookieJar()
    pro = urllib.request.HTTPCookieProcessor(cj)
    opener = urllib.request.build_opener(pro)
    header = []
    for key, value in head.items():
        elem = (key, value)
        header.append(elem)
    opener.addheaders = header
    return opener