# 装饰函数
def decre(func):
	def dec(x):
		print ("hello")
		func(x)
		print ("bye")
	return dec

@decre
def func(x):
	print (x)

func ('xixi')