import re

# get XSRF from web
def getXSRF(data):
    cer = re.compile('name=\"_xsrf\" value=\"(.*)\"', flags = 0)
    strlist = cer.findall(data)
    return strlist[0]

# get CSRF from web
def getCSRF(data):
    cer = re.compile('name=\"_csrf\" value=\"(.*)\"', flags = 0)
    strlist = cer.findall(data)
    return strlist[0]