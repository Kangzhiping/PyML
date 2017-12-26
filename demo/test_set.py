# test set
import platform

basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
print(basket)
print('orange' in basket)

a = set('abracadabra')
b = set('alacazam')
print (a) ; print (b)
print(a-b)
print(a|b)
print(a&b) # 集合a和b中都包含了的元素
print(a^b) ## 不同时包含于a和b的元素
print(platform)

