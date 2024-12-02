# 生成器调用
# import time
# def count_row():
#     num = 0
#     while num <= 100:
#         num += 20
#         yield num
# num = count_row()
#
# try:
#     while True:
#         print(next(num))
#         time.sleep(0.1)
# except StopIteration:
#     print('StopIteration')

# 内置函数重写
# class int(int):
#     def __new__(cls, *args, **kwargs):
#         res = super().__new__(cls, *args, **kwargs)
#         from random import random
#         if random() < 0.5:
#             res += 10
#         return res
#
# class A(object):
#     def __new__(cls, *args, **kwargs):
#         from random import random
#         if random() < 0.5:
#             return super().__new__(cls, *args, **kwargs)
#         else:
#             return None
#
# a = A()
# print(a is None)

# 切片slipce对象学习
# class test(object):
#     def __init__(self):
#         pass
#     def __getitem__(self, item):
#         print(item.start)
#         print(item.stop)
#         print(item.step)
# a = test()
# a['1':'test']

# 绑定在类下面的函数调用（类方法的调用实例）
class test_type(object):
    def __init__(self):
        pass
    def test_method(self):
        print('test_method')
    @staticmethod
    def test_static_method():
        print('test_static_method')
test_object = test_type()
test_object.test_method()
test_type.test_method(test_object)
print(type(test_object.test_method()))
print(type(test_object.test_method))
print(type(test_type.test_method))
print(type(test_type.test_static_method))
print(type(test_type().test_static_method))
