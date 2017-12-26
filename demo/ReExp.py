#!/usr/bin/python
import re

line = "Cats are smarter than dogs"

#re.M 使$匹配一行（串的不只是端部）的尾部，使^匹配的行（串不只是开始）的开始
#re.I IGNORECASE  Perform case-insensitive matching.
#re.S 使一个句号（点）匹配任何字符，包括换行符
#re.U 根据Unicode字符集解释的字母。这个标志会影响w, W, , B的行为。
#re.X 许可证“cuter”正则表达式语法。它忽略空格（除了一组[]或当用一个反斜杠转义内），并把转义＃作为注释标记

matchObj = re.match( r'(.*) are (.*?) .*', line, re.M|re.I)

if matchObj:
   print ("matchObj.group() : ", matchObj.group())
   print ("matchObj.group(1) : ", matchObj.group(1))
   print ("matchObj.group(2) : ", matchObj.group(2))
else:
   print ("No match!!")

# search 找第一次出现的地方
searchObj = re.search( r'(.*) are (.*?) .*', line, re.M|re.I)

if searchObj:
   print ("searchObj.group() : ", searchObj.group())
   print ("searchObj.group(1) : ", searchObj.group(1))
   print ("searchObj.group(2) : ", searchObj.group(2))
else:
   print ("Nothing found!!")

# match 只基于开头匹配
matchObj = re.match( r'dogs', line, re.M|re.I)
if matchObj:
   print ("match --> matchObj.group() : ", matchObj.group())
else:
   print ("No match!!")

# search 基于所有查找
searchObj = re.search( r'dogs', line, re.M|re.I)
if searchObj:
   print ("search --> searchObj.group() : ", searchObj.group())
else:
   print ("Nothing found!!")

phone = "2004-959-559 # This is Phone Number"

# Delete Python-style comments
num = re.sub(r'#.*$', "", phone)
print ("Phone Num : ", num)

# Remove anything other than digits
num = re.sub(r'\D', "", phone)
print ("Phone Num : ", num)