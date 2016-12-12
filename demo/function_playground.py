
def test():
    while True:
        x = (yield)
        y = x ** 2
        yield y

bla = test()
next(bla)
for i in range(2, 10):
    print(bla.send(i))
    next(bla)
