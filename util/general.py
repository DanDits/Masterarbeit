
# Decorater to start a coroutine automatically without having to call next() explicitly
def coroutine(func):
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        next(cr)
        return cr
    return start
