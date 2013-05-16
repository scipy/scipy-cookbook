import sys
import time
import threading
from itertools import izip, count

def foreach(f,l,threads=3,return_=False):
    """
    Apply f to each element of l, in parallel
    """

    if threads>1:
        iteratorlock = threading.Lock()
        exceptions = []
        if return_:
            n = 0
            d = {}
            i = izip(count(),l.__iter__())
        else:
            i = l.__iter__()


        def runall():
            while True:
                iteratorlock.acquire()
                try:
                    try:
                        if exceptions:
                            return
                        v = i.next()
                    finally:
                        iteratorlock.release()
                except StopIteration:
                    return
                try:
                    if return_:
                        n,x = v
                        d[n] = f(x)
                    else:
                        f(v)
                except:
                    e = sys.exc_info()
                    iteratorlock.acquire()
                    try:
                        exceptions.append(e)
                    finally:
                        iteratorlock.release()
        
        threadlist = [threading.Thread(target=runall) for j in xrange(threads)]
        for t in threadlist:
            t.start()
        for t in threadlist:
            t.join()
        if exceptions:
            a, b, c = exceptions[0]
            raise a, b, c
        if return_:
            r = d.items()
            r.sort()
            return [v for (n,v) in r]
    else:
        if return_:
            return [f(v) for v in l]
        else:
            for v in l:
                f(v)
            return

def parallel_map(f,l,threads=3):
    return foreach(f,l,threads=threads,return_=True)

if __name__=='__main__':
    def f(x):
        print x
        time.sleep(0.5)
    foreach(f,range(10))
    def g(x):
        time.sleep(0.5)
        print x
        raise ValueError, x
        time.sleep(0.5)
    foreach(g,range(10))
