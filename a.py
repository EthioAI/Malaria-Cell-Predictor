a = [1,2,3, 7, 8]
b = [4,5,6]

if len(a) > len(b):
    a = a[len(b):]
    print(a)