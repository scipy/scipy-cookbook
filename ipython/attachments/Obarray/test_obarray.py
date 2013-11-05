import numpy as np
from numpy.testing import *
import obarray

class TestObarray(NumpyTestCase):
    def setUp(self):
        
        class Foo:
            def __init__(self, a, b):
                self.a = a
                self.b = b
            def __str__(self):
                return "<Foo a=%s b=%s>" % (self.a, self.b)

        dtype = [("a",np.int),
                 ("b",np.float)]
        FooArray = obarray.make_obarray(Foo, dtype)

        A = FooArray([Foo(0,0.1),Foo(1,1.2),Foo(2,2.1),Foo(3,3.3)])

        self.Foo = Foo
        self.dtype = dtype
        self.FooArray = FooArray
        self.A = A

    def check_scalar_indexing(self):
        f = self.A[0]
        self.assert_(isinstance(f,self.Foo))
        assert_equal((f.a, f.b), (0,0.1))

    def check_subarray_indexing(self):
        A = self.A
        assert_equal(type(A[::2]),type(A))
        assert_equal(type(A[[True,False,True,True]]),type(A))
        assert_equal(type(A[[0,0,0]]),type(A))

    def check_scalar_assignment(self):
        Foo = self.Foo
        f = Foo(-1,-1.1)
        self.A[0] = Foo(-1,-1.1)
        assert_equal(self.A[0].a, f.a)
        assert_equal(self.A[0].b, f.b)

    def check_vector_assignment(self):
        self.A[::2] = self.A[1::2]
        assert_equal(self.A[0].a, self.A[1].a)
        assert_equal(self.A[0].b, self.A[1].b)


if __name__=='__main__':
    NumpyTest().run()


