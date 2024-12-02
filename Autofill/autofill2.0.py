import time
import tkinter as tk
import datetime as dt
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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

def button_commond(input_info, output_info):
    input_url, input_time, input_text = input_info
    input_datas = [str_ for str_ in input_text.get(1.0, 'end').split('\n') if str_]
    auto_fill(input_url.get(), input_time.get(), input_datas)
    output_info.set('output:' + '提交成功')
    
def auto_fill(input_url, input_time, input_text:list):
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
    driver.get(input_url)
    # 获取时间
    setclock = input_time.split(':')
    current_time = dt.datetime.now()
    execute_time = dt.datetime(current_time.year, current_time.month, current_time.day,
                               int(setclock[0]), int(setclock[1]), int(setclock[2]))
    wait_time = (execute_time - current_time).total_seconds()
    if wait_time >= 0 :
        time.sleep(wait_time - 1)
        while True:
            if dt.datetime.now() >= execute_time:
                break
        driver.refresh()
        locator = (By.CSS_SELECTOR, 'div[id=divQuestion]')
        WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located(locator))
    # 寻找元素并填写
    input_elements = driver.find_elements(By.XPATH, '//div[@id="divQuestion"]//input[@type="text"]')
    min_num = len(input_text) if len(input_text) <= len(input_elements) else len(input_elements)
    for idx in range(min_num):
        input_elements[idx].send_keys(input_text[idx])
    driver.find_element(By.XPATH, '//*[@id="ctlNext"]').click()
    # 销毁
    driver.quit()

if __name__ == '__main__':
    win = create_win('300x300', 'autofill')
    # 设置输入窗口
    input_info = set_input(win)
    # 设置输出文本
    output_info = tk.StringVar()
    label = tk.Label(win, textvariable=output_info, font=('楷体', 13), wraplength=300)
    label.place(x=0, y=250)
    # 运行
    try:
        button = tk.Button(win, text='确定', font=('楷体', 13), 
                       command=lambda:button_commond(input_info, output_info)).place(x=125, y=200)
        output_info.set('output:' + '请填写')
    except Exception as e:
        output_info.set('output:' + str(e))
    # 显示
    win.mainloop()