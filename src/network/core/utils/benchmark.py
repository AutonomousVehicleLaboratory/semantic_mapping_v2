import time


def timer(func):
    """A decorator for benchmarking the runtime of a function"""

    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print("%s takes %fs" % (func.__name__, (t2 - t1)))
        return res

    return wrapper


def model_timer(model, *arg, **kwargs):
    """
    Benchmark the inference time of a model
    """
    t1 = time.time()
    res = model(*arg, **kwargs)
    t2 = time.time()
    print("%s takes %fs" % (type(model).__name__, (t2 - t1)))
    return res
