#! /usr/lib/...
#
import os
import os.path
import pickle

d1 = {'a':1,'b':2,'c':3}
filename = 'C:\\Users\\IBM_ADMIN\\MikeFile\\Python\\test.txt'
print (os.path.isfile(filename)) #if file existing then true, otherwise false
print (d1)
#if os.path.isfile(filename):
f1 = open(filename,'ab+')
pickle.dump(d1,f1) #将对象用流的方式写进file.
f1.flush()
f1.close()

f2 = open(filename,'rb')
d2 = pickle.load(f2)
print(d2)
f2.close()

f3 = open(filename,'a+')
while True:
    line = input('Enter somethong>')
    if line == 'q' or line == 'quit':
        break
    f3.write(line + '\n')

f3.close()