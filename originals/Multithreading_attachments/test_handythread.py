#!/usr/bin/env python
import unittest

import handythread


class HandythreadTest(unittest.TestCase):
    def test_coverage(self):
        d = {}
        l = range(100)
        def f(x):
            d[x]=x**2
        handythread.foreach(f, l)
        for i in l:
            self.assertEqual(d[i],i**2)

    def test_return(self):
        l = range(100)
        r = handythread.foreach(lambda x: x**2, l, return_=True)
        for i in range(len(l)):
            self.assertEqual(l[i]**2,r[i])

    def test_return_1(self):
        l = range(100)
        r = handythread.foreach(lambda x: x**2, l, return_=True, threads=1)
        for i in range(len(l)):
            self.assertEqual(l[i]**2,r[i])

    def test_parallel_map(self):
        l = range(100)
        r = handythread.parallel_map(lambda x: x**2, l)
        for i in range(len(l)):
            self.assertEqual(l[i]**2,r[i])


if __name__=='__main__':
    unittest.main()
