# 事件循环只能执行task对象
# 协程依旧是单进程单线程的运行模式
# 适用于包含大量等待时间的程序, 如网络请求
'''
asyncio.run()函数:
1.传入协程对象, 将其包装成task, 创建event loop并注册该task, 开始event loop的运行

交回控制权给event loop的方式有两种:
1.await关键字交回
2.task执行完毕后交回

await关键字详解:
1.await + 协程对象: 协程对象被包装成task对象(task1)并被添加到event loop中
同时通知event loop现在执行的task需要等到await后task1执行完毕后才能执行, 建立依赖关系
交还现在执行task的控制权给event loop并暂停执行该task
event loop再次安排该task运行时, 返回依赖task1运行结果
2.await + task: 省略包装成task并注册的过程
3.await + future: 告诉event loop需要等待future的所有task都被执行完毕才会继续执行
交还现在执行task的控制权给event loop并暂停执行该task
event loop再次安排该task运行时, 返回依赖所有task运行结果的列表

create_task函数:
1.传入协程对象, 将其包装成task, 并将task注册到event loop中

gather函数:
1.传入task则返回future对象
2.传入协程对象则会将其包装成task注册到event loop中
'''
# import asyncio
# import time

# async def delay(t, str_):
#     print(str_, 1, sep=' ')
#     await asyncio.sleep(t)
#     print(str_, 2, sep=' ')

# async def main():
#     print(time.strftime('%X'))
#     await asyncio.gather(*(delay(i+1, f'task{i+1}') for i in range(2)))
#     print(time.strftime('%X'))

# asyncio.run(main())

'''
线程池:
future对象使用result方法时会等待线程结束
pool.map方法返回一个包含所有结果的迭代器
pool.submit方法返回future对象, 其组成的迭代对象可使用as_completed方法
future.result()返回结果
future.exception()获取线程运行时产生的异常
进程池用法与其一致
'''
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def test(args):
    name, t = args
    print(f'task:{name} - 1')
    time.sleep(t)
    print(f'task:{name} - 2')
    return f'task{name} completed'
    
with ThreadPoolExecutor() as pool:
    results1 = pool.map(test, [(i, 1) for i in range(5)])
    for result1 in results1:
        print(result1)
    results2 = [pool.submit(test, (i, 1)) for i in range(5)]
    for result2 in as_completed(results2):
        print(result2.result())
