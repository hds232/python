import asyncio
import threading
import tkinter as tk
import datetime as dt
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class is_element_ok(object):
    def __init__(self, selector):
        self.selector = selector
    def __call__(self, driver):
        driver.refresh()
        ele = EC.presence_of_all_elements_located(self.selector)(driver)
        if ele:
            return ele
        else:
            return False

def create_win(size, title):
    win = tk.Tk()
    win.title(title)
    win.geometry(size)
    win.resizable(False, False)
    return win

def set_input(win:tk.Tk):
    tk.Label(win, text='网址:', font=('楷体', 13)).place(x=0, y=0)
    input_url = tk.Entry(win, width=30)
    input_url.place(x=50, y=0)
    tk.Label(win, text='时间:', font=('楷体', 13)).place(x=0, y=25)
    input_time = tk.Entry(win, width=30)
    input_time.place(x=50, y=25)
    tk.Label(win, text='输入:', font=('楷体', 13)).place(x=0, y=50)
    input_text = tk.Text(win, width=34, height=10)
    input_text.place(x=50, y=50)
    return input_url, input_time, input_text

async def auto_fill(input_url, input_time, input_text):
    if not input_url:
        raise ValueError('未输入网址')
    # 设置浏览器选项
    option = webdriver.ChromeOptions()
    option.add_argument('--disable-automation')
    option.add_experimental_option('excludeSwitches', ['enable-automation'])
    option.add_experimental_option('useAutomationExtension', False)
    option.add_experimental_option('excludeSwitches', ['enable-logging'])
    driver = webdriver.Chrome(options=option)
    driver.execute_cdp_cmd(
        cmd='Page.addScriptToEvaluateOnNewDocument',
        cmd_args={'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'}
    )
    try:
        driver.get(input_url)
    except:
        raise ValueError('输入网址不正确')
    # 获取时间
    if input_time:
        setclock = input_time.split(':')
        if len(setclock) != 3:
            raise ValueError('时间格式错误')
        current_time = dt.datetime.now()
        execute_time = dt.datetime(current_time.year, current_time.month, current_time.day,
                                    int(setclock[0]), int(setclock[1]), int(setclock[2]))
        wait_time = (execute_time - current_time).total_seconds()
        if wait_time >= 0:
            await asyncio.sleep(wait_time - 1)
            locator = (By.CSS_SELECTOR, 'div[id=divQuestion]')
            WebDriverWait(driver, 11).until(is_element_ok(locator))
    # 寻找元素并填写
    input_elements = driver.find_elements(By.XPATH, '//div[@id="divQuestion"]//input')
    min_num = len(input_text) if len(input_text) <= len(input_elements) else len(input_elements)
    for idx in range(min_num):
        input_elements[idx].send_keys(input_text[idx])
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="ctlNext"]'))).click()
    # 等待问卷提交完毕
    await asyncio.sleep(3)
    # 销毁
    driver.quit()

def run_auto_fill(input_url, input_time, input_text, output_info):
    try:
        asyncio.run(auto_fill(input_url, input_time, input_text))
        output_info.set('output:提交成功')
    except Exception as e:
        output_info.set('output:' + str(e))

def button_commond(input_info, output_info):
    input_url, input_time, input_text = input_info
    input_datas = [str_ for str_ in input_text.get(1.0, 'end').split('\n') if str_]
    # 在新线程中运行 auto_fill
    thread = threading.Thread(target=run_auto_fill, args=(input_url.get(), input_time.get(), input_datas, output_info))
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    win = create_win('300x350', 'autofill')
    # 设置输入窗口
    input_info = set_input(win)
    # 设置输出文本
    output_info = tk.StringVar()
    tk.Label(win, textvariable=output_info, font=('仿宋', 13), wraplength=300).place(x=0, y=230)
    # 运行
    output_info.set('output:请填写')
    tk.Button(win, text='确定', font=('楷体', 13),
              command=lambda: button_commond(input_info, output_info)).place(x=125, y=200)
    # 显示
    win.mainloop()