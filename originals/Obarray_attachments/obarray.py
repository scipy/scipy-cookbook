import numpy as np
def make_obarray(klass, dtype):
    class Obarray(np.ndarray):
        def __new__(cls, obj):
            A = np.array(obj,dtype=np.object)
            N = np.empty(shape=A.shape, dtype=dtype)
            for idx in np.ndindex(A.shape):
                for name, type in dtype:
                    N[name][idx] = type(getattr(A[idx],name))
            return N.view(cls)
        def __getitem__(self, idx):
            V = np.ndarray.__getitem__(self,idx)
            if np.isscalar(V):
                kwargs = {}
                for i, (name, type) in enumerate(dtype):
                     kwargs[name] = V[i]
                return klass(**kwargs)
            else:
                return V
        def __setitem__(self, idx, value):
            if isinstance(value, klass):
                value = tuple(getattr(value, name) for name, type in dtype)
            # FIXME: treat lists of lists and whatnot as arrays
            return np.ndarray.__setitem__(self, idx, value)
    return Obarray


