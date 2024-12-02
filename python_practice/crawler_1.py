import time
import requests
import tkinter as tk
from bs4 import BeautifulSoup

win = tk.Tk()
win.title('奥运奖牌榜')
win.geometry('600x500')

scrollbar = tk.Scrollbar(win)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
mylist = tk.Listbox(win, yscrollcommand=scrollbar.set, font=('宋体', 17))

heads = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36 '
                       'SLBrowser/9.0.3.5211 SLBChan/105'}
response = requests.get('https://cn.bing.com/sportsdetails?'
                        'q=%E5%A5%A5%E8%BF%90%E9%87%91%E7%89%8C%E6%A6%9C'
                        '&sport=Olympics&TimezoneId=China%20Standard%20Time'
                        '&intent=Standings&isolympics=True&segment=sports&isl2=true'
                        '&form=QBLH&', headers=heads)
soup = BeautifulSoup(response.text, 'html.parser')

all_titles = soup.find_all('span', attrs={'class':'bsp_row_teamname'})
all_nums = soup.find_all('td', attrs={'class':'colVal'})
list_num = [int(x.string) for x in all_nums]
all_1_3 = soup.find_all('div', attrs={'class':'bsp_oly_med_team_name'})
all_1_3_total_num = soup.find_all('div', attrs={'class':'bsp_oly_med_total'})
list_total_num = [int(x.string) for x in all_1_3_total_num]
all_1_3_num = soup.find_all('div', attrs={'class':'bsp_oly_med_count'})
list_1_3_num = [int(x.string) for x in all_1_3_num]

t = time.localtime()
mylist.insert(tk.END, f'    截止到{t.tm_year}年{t.tm_mon}月{t.tm_mday}日'+
                      f'{t.tm_hour}时{t.tm_min}分{t.tm_sec}秒')
mylist.insert(tk.END, '    数据来源于https://cn.bing.com/sportsdetails')

for name,total_num,j in zip(all_1_3, list_total_num, range(int(len(list_1_3_num)/3))):
    temp_list = list_1_3_num[3*j:3*j+3]
    temp_list.append(total_num)
    mylist.insert(tk.END, f'{name.string}:{temp_list}')
for title,i in zip(all_titles,range(int(len(list_num)/4))):
    mylist.insert(tk.END, f'{title.string}:{list_num[4*i:4*i+4]}')

mylist.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
scrollbar.config(command=mylist.yview)
win.mainloop()