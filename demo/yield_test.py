def yieldTest(x):
    while x>0:
        yield x
        x -= 1
y = yieldTest(10)
print(type(y))

for z in y:
    print(z)

