
import ctypes
import numpy as np
lib = ctypes.cdll.LoadLibrary('lib_demo_class.so')

class MyClass(object):
    def __init__(self, p: int = None, q: int = None):

        # define input types
        lib.NewMyClass.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.DefaultMyClass.argtypes = []
        lib.GetP.argtypes = [ctypes.c_void_p]
        lib.GetQ.argtypes = [ctypes.c_void_p]
        lib.Calc.argtypes = [ctypes.c_void_p]
        lib.Delete.argtypes = [ctypes.c_void_p]
        lib.SetArray.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        lib.GetArray.argtypes = [ctypes.c_void_p]

        # define output types
        lib.NewMyClass.restype = ctypes.c_void_p
        lib.DefaultMyClass.restype = ctypes.c_void_p
        lib.GetP.restype = ctypes.c_int
        lib.GetQ.restype = ctypes.c_int
        lib.Calc.restype = ctypes.c_int
        lib.Delete.restype = ctypes.c_void_p
        lib.SetArray.restype = ctypes.c_void_p
        lib.GetArray.restype = ctypes.POINTER(ctypes.c_float * 3)

        if p is None:
            if q is None:
                self.obj = lib.DefaultMyClass(p, q)
            else:
                raise Exception("Pass p and q or neither")
        else:
            self.obj = lib.NewMyClass(p, q)

    def get_p(self):
        return lib.GetP(self.obj)

    def get_q(self):
        return lib.GetQ(self.obj)

    def calc(self):
        return lib.Calc(self.obj)

    def set_array(self, a: float, b: float, c: float):
        return lib.SetArray(self.obj, a, b, c)

    def get_array(self):
        return np.array(lib.GetArray(self.obj).contents)

    def __del__(self):
        return lib.Delete(self.obj)


if __name__ == "__main__":
    myClass = MyClass(4, 5)
    print(myClass.get_p())
    print(myClass.get_q())
    print(myClass.calc())
    myClass.set_array(1., 4.3, 7.2)
    print(myClass.get_array())

    myDefaultClass = MyClass()
    print(myDefaultClass.get_p())
    print(myDefaultClass.get_q())
    print(myDefaultClass.calc())
