import time
# 装饰器的用法
def timeit(f):
    def wrapper(x):
        stat = time.time()
        ret = f(x)
        print(time.time() - stat)
        return ret
    return wrapper
# 等价于 myfunc = timeit(myfunc)
@timeit
def my_func(x):
    time.sleep(x)
my_func(0.1)

# 带参数的装饰器
def timeit_params(count:int|float):
    print(f'test count:{count}')
    def timeit(fun):
        def wrapper(*args, **kwargs):
            stat_time = time.time()
            result = fun(*args, **kwargs)
            end_time = time.time()
            print(f'test time:{end_time - stat_time}')
            return result
        return wrapper
    return timeit
@timeit_params(1)
def test_fun(time_num:float):
    time.sleep(time_num)
test_fun(2)

# 打包解包操作
def unpack_test(num1, num2):
    print(f'return:{num1 + num2}')
test_data1 = [1, 2]
test_data2 = {'num1':2, 'num2':3}
unpack_test(*test_data1)
unpack_test(**test_data2)
print(unpack_test.__name__)


# 生成器的用法
# class Node:
#     def __init__(self, name:str):
#         self.name = name
#         self.next = None
#     #下面也可定义一个__iter__返回self本身，再定义一个__next__函数返回next函数的返回值
#     def __iter__(self):
#         node = self
#         while node is not None:
#             yield node
#             node = node.next
# #生成器函数返回一个生成器对象，使用next函数调用生成器对象会返回yield后面的值。
# #程序运行到yield的时候暂停，等待下一次调用next函数时继续运行并依次循环
#
# def creatrnode(*args, **kwargs):
#     name_list = list(args)
#     if kwargs is not None:
#         name_list_expand = list(kwargs.values())
#         name_list.extend(name_list_expand)
#     node_list = []
#     for name in name_list:
#         node_list.append(Node(name))
#     for i in range(len(node_list) - 1):
#         node_list[i].next = node_list[i + 1]
#     return node_list
#
# node_list = creatrnode('node1', 'node2', node3='node3', node4='node4')
# node1 = node_list[0]
# #for循环首先对In后对象调用iter函数，再在每次循环中调用next函数
# for node in node1:
#     print(node.name)