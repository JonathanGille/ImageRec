def t():
    x = 2
    y = 3
    z = 4
    return x,y,z

n = [t(),t(),t()]
print(n)
for x,y,z in n:
    print(x,y,z)