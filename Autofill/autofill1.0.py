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
    tk.Label(win, text='姓名:', font=('楷体', 13)).place(x=0, y=0)
    input_name = tk.Entry(win, width=30)
    input_name.place(x=50, y=0)
    tk.Label(win, text='学号:', font=('楷体', 13)).place(x=0, y=25)
    input_num = tk.Entry(win, width=30)
    input_num.place(x=50, y=25)
    tk.Label(win, text='网址:', font=('楷体', 13)).place(x=0, y=50)
    input_url = tk.Entry(win, width=30)
    input_url.place(x=50, y=50)
    tk.Label(win, text='时间:', font=('楷体', 13)).place(x=0, y=75)
    input_time = tk.Entry(win, width=30)
    input_time.place(x=50, y=75)
    return input_name, input_num, input_url, input_time

def button_commond(input_info, output_info):
    input_datas = []
    for info in input_info:
        info.configure(state='readonly')
        input_datas.append(info.get())
    auto_fill(input_datas)
    output_info.set('output:' + '提交成功')
    

def auto_fill(input_datas):
    # 设置浏览器选项
    option = webdriver.ChromeOptions()
    option.add_argument('--disable-automation')
    option.add_experimental_option('excludeSwitches', ['enable-automation'])
    option.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome(options=option)
    driver.execute_cdp_cmd(
        cmd='Page.addScriptToEvaluateOnNewDocument',
        cmd_args={'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'}
    )
    driver.get(input_datas[2])

    # 获取时间
    setclock = input_datas[3].split(':')
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
        locator = (By.CSS_SELECTOR, 'input[id=q1]')
        WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located(locator))

    driver.find_element(By.CSS_SELECTOR, 'input[id=q1]').send_keys(input_datas[0])
    driver.find_element(By.CSS_SELECTOR, 'input[id=q2]').send_keys(input_datas[1])
    driver.find_element(By.XPATH, '//*[@id="ctlNext"]').click()

    driver.quit()

if __name__ == '__main__':
    win = create_win('300x300', 'autofill')
    
    input_info = set_input(win)
    button = tk.Button(win, text='确定', font=('楷体', 13), 
                       command=lambda:button_commond(input_info)).place(x=125, y=100)
    output_info = tk.StringVar()
    label = tk.Label(win, textvariable=output_info, font=('楷体', 13), wraplength=300)
    label.place(x=0, y=150)
    try:
        button = tk.Button(win, text='确定', font=('楷体', 13), 
                       command=lambda:button_commond(input_info, output_info)).place(x=125, y=100)
        output_info.set('output:' + '请填写')
    except Exception as e:
        output_info.set('output:' + str(e))
    
    win.mainloop()