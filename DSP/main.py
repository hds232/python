import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

url = 'https://www.icourse163.org/'
course_info = 'spoc/learn/UESTC-1471648183'
homework_idx = 7
review_num = 29
comment = '答案正确, 过程详细, 完美契合得分点'

def create_driver():
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
    "source": """
        Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined
        })"""})
    driver.get(url)
    return driver

def get_cookies(driver: webdriver.Chrome):
    cookies = driver.get_cookies()
    with open('cookies.json', 'w', encoding='utf-8') as f:
        json.dump(cookies, f)

def load_cookies(driver: webdriver.Chrome):
    with open('cookies.json', 'r', encoding='utf-8') as f:
        cookies = json.load(f)
    for cookie in cookies:
        driver.add_cookie(cookie)

def login(driver: webdriver.Chrome):
    driver.implicitly_wait(10)
    driver.find_element(By.XPATH, 
                        '//div[@class="_3uWA6" and @role="button"]').click()
    while True:
        match (ch := input('请登录:(yes/no/init/cookie)')):
            case 'yes': break
            case 'init': 
                get_cookies(driver)
                break
            case 'cookie':
                driver.delete_all_cookies()
                load_cookies(driver)
                driver.refresh()
                driver.implicitly_wait(10)
                break
            case 'no':
                driver.quit()
                exit()
            case _: print('输入错误，请重新输入')
    driver.implicitly_wait(10)

def enter_homework_page(driver: webdriver.Chrome):
    action = ActionChains(driver)
    driver.implicitly_wait(10)
    # driver.find_element(By.XPATH, '//span[@class="ux-btn  ux-btn- ux-btn- "]').click()
    # driver.implicitly_wait(10)
    elem = driver.find_element(By.XPATH, '//span[@class="ant-menu-title-content"]' +
                        '//a[text()="测验与作业"]')
    action.move_to_element(elem).perform()
    action.click().perform()

def find_homework_list(driver: webdriver.Chrome) -> list:
    driver.implicitly_wait(10)
    homework_list = driver.find_elements(By.XPATH, 
            '//div[@class="m-learnbox"]//' + 
            'a[@class="j-quizBtn u-btn u-btn-default f-fr" and text()="前往作业"]')
    return homework_list

def enter_peer_assessed_assignment(driver: webdriver.Chrome):
    elem = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, 
            '//div[@class="tasklist j-tasklist"]//a[@data-status="1"]'))
    )
    elem.click()

def find_all_options(driver: webdriver.Chrome) -> tuple:
    score_options_list = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.XPATH, 
                '//div[@class="m-homework u-learn-modulewidth"]' + 
                '//div[@class="s"]//label[last()]//input[last()]'))
    )
    text_options_list = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.XPATH, 
            '//div[@class="m-homework u-learn-modulewidth"]//textarea'))
    )
    return score_options_list, text_options_list

def enter_info(score_options_list, text_options_list):
    for i in range(len(score_options_list)):
        score_options_list[i].click()
    for i in range(len(text_options_list)):
        text_options_list[i].send_keys(comment)

def run_enter_info_iter(driver: webdriver.Chrome):
    for _ in range(review_num):
        driver.refresh()
        driver.implicitly_wait(10)
        enter_peer_assessed_assignment(driver)
        time.sleep(3)
        score_options_list, text_options_list = find_all_options(driver)
        enter_info(score_options_list, text_options_list)
        time.sleep(3)
        driver.find_element(By.XPATH, 
            '//a[@class="u-btn u-btn-default f-fl j-submitbtn"]').click()
        driver.implicitly_wait(10)
        driver.find_element(By.XPATH, '//a[@class="j-gotonext"]').click()

def main():
    # 获得课程网址
    course_url = url + course_info
    # 打开浏览器并登录
    driver = create_driver()
    login(driver)
    driver.get(course_url)
    # 寻找作业列表
    enter_homework_page(driver)
    homework_list = find_homework_list(driver)
    homework_list[homework_idx].click()
    # 寻找互评作业并进入
    run_enter_info_iter(driver)
    # 销毁浏览器
    time.sleep(3)
    driver.quit()

if __name__ == '__main__':
    main()