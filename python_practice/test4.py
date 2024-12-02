# """
# import关键字导入pkg时会将调用路径上的pkg的__init__都调用一遍
# 在代码中实现层级调用时需要在__init__文件中import下级包
# 否则只能在import中才能实现层级调用(import testpkg.subtestpkg)
# __all__的定义规定了import *的行为
# """
# import testpkg.subtestpkg
# import pprint
#
# pprint.pprint(dir(testpkg))
# print(testpkg.subtestpkg)
# print(testpkg)

# 元类，实现Test_Class = Test_Type(...)替代type元类的过程
# 在建立类的过程中添加自己的处理过程
# 调用父类type时最好不要用super
# class Test_Type(type):
#     def __new__(cls, name, bases, dicts):
#         pass
#         return type.__new__(cls, name, bases, dicts)
#     def __init__(self, name, bases, dicts):
#         pass
#         return type.__init__(self, name, bases, dicts)
#     def __call__(self, *args, **kwargs):
#         pass
#         return type.__call__(self, *args, **kwargs)
# # 设置元类不会影响创造实例，只会影响这个类本身的建立过程
# class Test_Class(metaclass=Test_Type):
#     def __init__(self):
#         print('test_class_init')
# o = Test_Class()

# method类详解
# import pprint as p

# class Test_Class(object):
#     Status = True
#     def __init__(self):
#         self.name = 'test'
#     # 定义在类中的方法本身是一个function
#     def test_method(self):
#         print(self.name)
# tst_obj = Test_Class()
# p.pprint(Test_Class.__dict__)
# p.pprint(tst_obj.__dict__)
# 描述器:
# 对象调用test_method时，先检查tst_obj的__dict__中是否有test_method
# 没有则在其type中找，找到之后调用function的__get__方法，得到一个特殊的method对象
# tst_obj.f = lambda self, x:print(self.name, x)
# try:
#     tst_obj.f('x')
# except Exception as e:
#     print(e)

'''
对象属性的调用和设置:
__setattr__方法: 在任意位置(包括类中self.data = val)设置对象属性时都会被调用
__getattr__方法: 当访问一个对象和对象的type都不存在的属性时调用
__getattribute__方法: 访问对象属性时(无论是否存在)都会被调用
'''
# 代码将所有的类属性绑定到_attr中，实现所有对象的统一更改
class Test(object):
    _attr = {}
    def __init__(self):
        self.data = 123
    def __setattr__(self, name, val):
        self._attr[name] = val
    def __getattr__(self, name):
        if name not in self._attr:
            raise AttributeError
        return self._attr[name]
test1 = Test()
test2 = Test()
test1.data = 666
print(test2.data)
