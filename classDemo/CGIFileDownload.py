#!/usr/bin/python

# HTTP Header
print ("Content-Type:application/octet-stream; name=\"FileName\"")
print ("Content-Disposition: attachment; filename=\"FileName\" ")

# Actual File Content will go hear.
fo = open("foo.txt", "rb")

str = fo.read();
print (str)

# Close opend file
fo.close()