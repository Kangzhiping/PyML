#!/usr/bin/python
import os

print ("Content-type:text/html")
print ('<html>')
print ('<head>')
print ('<title>Hello Word - First CGI Program</title>')
print ('</head>')
print ('<body>')
print ('<h2>Hello Word! This is my first CGI program</h2>')
print ('</body>')
print ('</html>')

print ("<font size=+1>Environment</font><r>")
# 打印所有的环境变量参数
for param in os.environ.keys():
#  print ("<b>%20s</b>: %s<r>" % (param, os.environ[param]))
  print ("<b>%20s</b>: %s" % (param, os.environ[param]))