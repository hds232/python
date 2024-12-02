# a = [1, 2, 3]
# b = [1, 2, 3]
# print(a == b)
# print(a is b)

# 自定义类and和&的区别
class test_class(object):
    def __init__(self, data):
        self.data = data
    def __bool__(self):
        if self.data == []:
            return False
        else:
            return True
    def __eq__(self, other):
        return self.data == other.data
    def __hash__(self):
        return hash(self.data)

test_data = [1, 2, 3]
test_obj1 = test_class(test_data)
test_obj2 = test_class([])
try:
    print(test_obj1 and test_obj2)
except Exception as e:
    print('test1 error')
try:
    '''
    自定义类中未定义__and__方法，故此段代码报错
    '''
    print(test_obj1 & test_obj2)
except Exception as e:
    print('test2 error')
if test_obj1:
    '''
    检查__bool__方法的返回
    '''
    print('__bool__ return Ture')
# and 返回前后两个对象中的一个，若两个对象均不是Ture, False, None
# 中的任意一个，则返回and后的对象.
print(1 and 2)
print(test_obj1 and test_obj2)
# or 返回前后两个对象中的一个，若两个对象都是False或None，则返回False,
# 否则返回or前的对象
print(1 or 2)
print(test_obj1 or test_obj2)
print(None or False)
print('1' or '2')
print('2' or '1')
# not 运算符调用自定义对象的__bool__方法
print(not test_obj2)