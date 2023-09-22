import array
from datetime import datetime, timedelta
# 两种方式 sharedctypes 和 Manager
from multiprocessing import Manager
from multiprocessing import Process
from multiprocessing.sharedctypes import RawArray

size = 1000000


def tranverse(a):
    t = datetime.now()
    for i in range(size):
        a[i]
    print('elapsed %s' % (datetime.now() - t))


if __name__ == '__main__':
    a = array.array('i', [i for i in range(size)])
    print('test array')
    tranverse(a)
    a = {}
    for i in range(size):
        a[i] = i
    print('test dict')
    tranverse(a)

    manager = Manager()
    a = manager.list([i for i in range(size)])
    print('test shared manager list')
    tranverse(a)

    a = RawArray('i', [i for i in range(size)])
    print('test sharedctypes list in main process')
    tranverse(a)

    ps = [Process(target=tranverse, args=(a,)) for i in range(8)]
    print('test sharedctypes list in subprocess')
    for p in ps:
        p.start()
    for p in ps:
        p.join()
